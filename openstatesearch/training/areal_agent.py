from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openstatesearch.agent.harness import SearchHarness
from openstatesearch.agent.schemas import ActionValidationError, parse_action
from openstatesearch.agent.state import Budget, SearchState
from openstatesearch.retriever.hybrid import HybridRetriever
from openstatesearch.retriever.http import HttpRetriever
from openstatesearch.retriever.service import load_corpus
from openstatesearch.retriever.transformer_dense import TransformerDenseRetriever
from openstatesearch.rewards import (
    ABCCreditConfig,
    CreditTransition,
    EvidenceCreditTracker,
    RewardBreakdown,
    TrajectoryOutcome,
    compute_reward,
)

try:  # Kept optional so CPU-only unit tests can import this module.
    from areal.api import RolloutWorkflow
except ImportError:  # pragma: no cover - AReaL lives in its pinned training venv

    class RolloutWorkflow:  # type: ignore[no-redef]
        pass


SYSTEM_PROMPT = """You are a search policy. Respond with exactly one JSON action and no prose.
Allowed forms:
SEARCH {\"type\":\"SEARCH\",\"query\":str,\"target_constraint\":str}
OPEN {\"type\":\"OPEN\",\"doc_id\":str}
KEEP {\"type\":\"KEEP\",\"doc_id\":str,\"sent_ids\":[int],\"claim\":str,\"constraint_id\":str}
VERIFY {\"type\":\"VERIFY\",\"claim\":str,\"query\":str}
ANSWER {\"type\":\"ANSWER\",\"answer\":str,\"citations\":[{\"claim\":str,\"doc_id\":str,\"sent_ids\":[int]}]}
The harness owns budgets and state. Never invent doc_id or sent_id. Search until evidence is sufficient, then answer.
Loop rules: never repeat a SEARCH/VERIFY query or reopen an opened doc_id. If the last result has duplicate=true,
immediately choose a different query/doc/action. After OPEN, KEEP useful legal sentences before moving on. Do not
spend all turns searching: when remaining_turns <= 2, emit the best supported ANSWER using only opened citations.
The legal_action_space field is authoritative: OPEN only an openable_doc_id, cite only legal_citations when that
list is non-empty, and emit ANSWER immediately when must_answer=true. Keep every action concise. ANSWER must use
at most two citations and stay below 256 tokens."""

_AUDIT_THREAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class AgentEpisodeResult:
    breakdown: RewardBreakdown
    process_rewards: dict[str, float]
    transitions: tuple[CreditTransition, ...]


def _stable_sampling_seed(
    base_seed: int,
    version: int,
    trajectory_id: str,
    sample_idx: int,
    turn_index: int,
) -> int:
    material = (f"{base_seed}:{version}:{trajectory_id}:{sample_idx}:{turn_index}").encode("utf-8")
    # Keep the result independent of Python hash randomization and async request
    # order while staying within vLLM's non-negative signed seed range.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


def _append_audit_record(path: Path, payload: dict[str, Any], sample_size: int) -> bool:
    """Append one record while enforcing a restart-safe, per-step sample cap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize file descriptors within this process before acquiring flock;
    # Linux flock calls from separate threads can otherwise wait on each other
    # while occupying the entire asyncio worker pool.
    with _AUDIT_THREAD_LOCK:
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            step = payload["step"]
            count = 0
            for line in handle:
                try:
                    if json.loads(line).get("step") == step:
                        count += 1
                except json.JSONDecodeError:
                    # Preserve raw evidence even if a previous interrupted write is
                    # malformed; it must not silently consume the valid sample cap.
                    continue
            if count >= sample_size:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            return True


def legal_action_space(
    state: dict[str, Any],
    opened_doc_ids: list[str],
    remaining_turns: int,
    last_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive the mechanically legal frontier without consulting answer or evidence gold."""
    opened = set(opened_doc_ids)
    budget = state["budget"]
    openable = (
        [
            str(candidate["doc_id"])
            for candidate in state["candidate_pool"]
            if str(candidate["doc_id"]) not in opened
        ]
        if int(budget["open_left"]) > 0
        else []
    )
    legal_citations = [
        {
            "doc_id": str(evidence["doc_id"]),
            "sent_ids": list(evidence["sent_ids"]),
            "claim": str(evidence["claim"]),
        }
        for evidence in state["evidence"]
        if str(evidence["doc_id"]) in opened
    ]
    must_answer = remaining_turns <= 2 and bool(opened)
    allowed_types = []
    if int(budget["search_left"]) > 0 and not must_answer:
        allowed_types.extend(["SEARCH", "VERIFY"])
    if openable and not must_answer:
        allowed_types.append("OPEN")
    if (
        last_result
        and last_result.get("ok")
        and last_result.get("action") == "OPEN"
        and not last_result.get("duplicate")
    ):
        allowed_types.append("KEEP")
    if opened:
        allowed_types.append("ANSWER")
    return {
        "allowed_types": list(dict.fromkeys(allowed_types)),
        "openable_doc_ids": openable,
        "legal_citations": legal_citations,
        "must_answer": must_answer,
    }


def build_policy_input(
    harness: SearchHarness, remaining_turns: int, last_result: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the gold-free policy observation and its mechanically legal action frontier."""
    state = harness.state.observation()
    opened = sorted(harness.opened)
    return {
        "state": state,
        "opened_doc_ids": opened,
        "remaining_turns": remaining_turns,
        "legal_action_space": legal_action_space(state, opened, remaining_turns, last_result),
        "last_tool_result": last_result,
    }


def _refs(items: list[dict[str, Any]]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(item["doc_id"]), int(sent_id))
        for item in items
        for sent_id in item.get("sent_ids", [])
    )


def _policy_messages(
    policy_input: dict[str, Any], tokenizer: Any, max_prompt_tokens: int
) -> list[dict[str, str]]:
    """Bound an observation without changing any document or sentence identifiers."""
    value = json.loads(json.dumps(policy_input, ensure_ascii=False))

    def messages() -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(value, ensure_ascii=False)},
        ]

    def token_count() -> int:
        return len(
            tokenizer.apply_chat_template(
                messages(),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    last_payload = (value.get("last_tool_result") or {}).get("payload") or {}
    sentences = last_payload.get("sentences") or []
    for sentence in sentences:
        sentence["text"] = str(sentence.get("text", ""))[:768]
    candidates = value.get("state", {}).get("candidate_pool", [])
    for candidate in candidates:
        candidate["snippet"] = str(candidate.get("snippet", ""))[:256]

    while len(sentences) > 1 and token_count() > max_prompt_tokens:
        sentences.pop()
    if token_count() > max_prompt_tokens:
        for candidate in candidates:
            candidate["snippet"] = ""
    while sentences and token_count() > max_prompt_tokens:
        text = str(sentences[-1].get("text", ""))
        if not text:
            sentences.pop()
        else:
            sentences[-1]["text"] = text[: len(text) // 2]
    if token_count() > max_prompt_tokens:
        raise ValueError(
            "policy observation exceeds the configured context window after truncation"
        )
    return messages()


class OpenStateSearchAgent:
    """AReaL proxy-compatible agent; gold fields are used only after rollout."""

    _retriever: HybridRetriever | HttpRetriever | None = None

    @classmethod
    def _get_retriever(cls) -> HybridRetriever | HttpRetriever:
        if cls._retriever is None:
            service_url = os.environ.get("OSS36_RETRIEVER_URL")
            if service_url:
                cls._retriever = HttpRetriever(service_url)
                return cls._retriever
            corpus = os.environ.get("OSS36_CORPUS")
            model = os.environ.get("OSS36_DENSE_MODEL")
            index = os.environ.get("OSS36_DENSE_INDEX")
            if not corpus or not model or not index:
                raise RuntimeError(
                    "OSS36_CORPUS, OSS36_DENSE_MODEL and OSS36_DENSE_INDEX are required"
                )
            documents = load_corpus(corpus)
            dense = TransformerDenseRetriever(documents, model, index)
            cls._retriever = HybridRetriever(documents, dense=dense)
        return cls._retriever

    def __init__(
        self,
        tokenizer: Any,
        max_total_tokens: int,
        max_action_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        sampling_seed: int,
        abc_config: ABCCreditConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_total_tokens = max_total_tokens
        self.max_action_tokens = max_action_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.sampling_seed = sampling_seed
        self.abc_config = abc_config or ABCCreditConfig()

    async def run(
        self,
        data: dict[str, Any],
        client: Any,
        *,
        version: int = 0,
        sample_idx: int = 0,
    ) -> AgentEpisodeResult:
        """Run one episode while AReaL records every policy interaction."""
        question = str(data.get("question") or data.get("messages", [{}])[-1].get("content", ""))
        if not question:
            raise ValueError("rollout record requires question or messages[-1].content")
        trajectory_id = str(
            data.get("id")
            or data.get("_id")
            or hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
        )
        state = SearchState(
            question,
            constraints=list(data.get("constraints", [])),
            budget=Budget(search_left=4, open_left=4, token_left=8192),
        )
        harness = SearchHarness(state, self._get_retriever(), top_k=5)
        last_result: dict[str, Any] | None = None
        invalid_action_count = 0
        generated_tokens = 0
        credit_tracker = EvidenceCreditTracker(
            _refs(data.get("gold_evidence", data.get("supporting_facts", []))),
            self.abc_config,
        )
        process_rewards: dict[str, float] = {}
        transitions: list[CreditTransition] = []
        for turn_index in range(16):
            if harness.state.budget.token_left <= 0:
                break
            policy_input = build_policy_input(harness, 16 - turn_index, last_result)
            max_completion_tokens = min(self.max_action_tokens, harness.state.budget.token_left)
            response = await client.chat.completions.create(
                messages=_policy_messages(
                    policy_input,
                    self.tokenizer,
                    self.max_total_tokens - max_completion_tokens,
                ),
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=max_completion_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": self.top_k,
                    "seed": _stable_sampling_seed(
                        self.sampling_seed,
                        version,
                        trajectory_id,
                        sample_idx,
                        turn_index,
                    ),
                },
            )
            content = response.choices[0].message.content or ""
            completion_tokens = int(
                response.usage.completion_tokens if response.usage else len(content.split())
            )
            generated_tokens += completion_tokens
            harness.consume_tokens(completion_tokens)
            response_id = str(response.id)
            if not content.strip():
                # Immediate EOS is a real policy action, not an infrastructure
                # failure.  End the episode with an invalid-protocol reward so
                # GRPO can learn to suppress it without repeatedly resampling.
                invalid_action_count += 1
                last_result = {
                    "ok": False,
                    "action": "INVALID",
                    "error": "empty policy completion",
                }
                harness.events.append({"action": content, "result": last_result})
                transition = credit_tracker.score(None, last_result)
                transitions.append(transition)
                process_rewards[response_id] = transition.process_reward
                break
            try:
                action = parse_action(json.loads(content))
            except (json.JSONDecodeError, ActionValidationError, TypeError):
                invalid_action_count += 1
                last_result = {"ok": False, "action": "INVALID", "error": "invalid action JSON"}
                harness.events.append({"action": content, "result": last_result})
                transition = credit_tracker.score(None, last_result)
                transitions.append(transition)
                process_rewards[response_id] = transition.process_reward
                continue
            # HTTP retrieval is synchronous; keep it off AReaL's shared async loop.
            result = await asyncio.to_thread(harness.apply, action)
            last_result = result.to_dict()
            if not result.ok:
                invalid_action_count += 1
            transition = credit_tracker.score(action, last_result)
            transitions.append(transition)
            process_rewards[response_id] = transition.process_reward
            if harness.finished:
                break

        answer = harness.answer
        predicted_evidence = tuple(
            (evidence.doc_id, sent_id)
            for evidence in state.evidence
            for sent_id in evidence.sent_ids
        )
        citations = tuple(
            (citation.doc_id, sent_id)
            for citation in (answer.citations if answer else ())
            for sent_id in citation.sent_ids
        )
        returned = tuple(
            tuple(hit["doc_id"] for hit in event["result"].get("payload", {}).get("hits", []))
            for event in harness.events
            if isinstance(event.get("action"), dict)
            and event["action"].get("type") in {"SEARCH", "VERIFY"}
        )
        raw_answers = data.get("answers", [data.get("answer", "")])
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        outcome = TrajectoryOutcome(
            prediction=answer.answer if answer else "",
            references=tuple(str(value) for value in raw_answers),
            predicted_evidence=predicted_evidence,
            gold_evidence=_refs(data.get("gold_evidence", data.get("supporting_facts", []))),
            citations=citations,
            queries=tuple(state.query_history),
            returned_doc_ids=returned,
            search_count=len(state.query_history),
            open_count=len(harness.opened),
            generated_tokens=generated_tokens,
            invalid_action_count=invalid_action_count,
            # Recoverable malformed/tool actions no longer poison an otherwise
            # legal terminal answer. Their bounded penalty is accounted for by
            # compute_reward; an episode without an accepted answer stays hard
            # invalid, including an ANSWER with unkept/illegal citations.
            valid_tools=answer is not None,
            valid_citations=answer is not None,
        )
        phase = os.environ.get("OSS36_RL_PHASE", "A").upper()
        return AgentEpisodeResult(
            breakdown=compute_reward(outcome, phase),
            process_rewards=process_rewards,
            transitions=tuple(transitions),
        )


class OpenStateSearchWorkflow(RolloutWorkflow):
    """AReaL multi-turn workflow with trajectory-level reward propagation."""

    def __init__(
        self,
        gconfig: Any,
        tokenizer: Any,
        export_style: str = "individual",
        turn_discount: float = 1.0,
        credit_assignment: str = "terminal",
        abc_beta: float = 1.0,
        sampling_seed: int = 1,
    ) -> None:
        from areal.utils.hf_utils import load_hf_tokenizer

        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.export_style = export_style
        self.turn_discount = turn_discount
        if credit_assignment not in {"terminal", "abc"}:
            raise ValueError("credit_assignment must be 'terminal' or 'abc'")
        if abc_beta < 0.0:
            raise ValueError("abc_beta must be non-negative")
        self.credit_assignment = credit_assignment
        self.abc_beta = abc_beta
        self.max_total_tokens = int(gconfig.max_tokens)
        # ``max_new_tokens`` is the trajectory action envelope. SEARCH/OPEN/KEEP
        # naturally terminate well below it, while ANSWER may need the configured
        # 768-token ceiling for a complete JSON object plus citations. Capping all
        # turns at 256 truncated otherwise valid ANSWER objects mid-string.
        self.max_action_tokens = int(gconfig.max_new_tokens)
        self.agent = OpenStateSearchAgent(
            tokenizer=self.tokenizer,
            max_total_tokens=self.max_total_tokens,
            max_action_tokens=self.max_action_tokens,
            temperature=float(gconfig.temperature),
            top_p=float(gconfig.top_p),
            top_k=int(gconfig.top_k),
            sampling_seed=sampling_seed,
            abc_config=ABCCreditConfig(beta=abc_beta),
        )
        self.audit_path = Path(
            os.environ.get(
                "OSS36_REWARD_AUDIT",
                "artifacts/runtime/grpo_reward_audit.jsonl",
            )
        )
        self.audit_every_steps = 100
        self.audit_sample_size = 50

    async def _record_audit(
        self,
        version: int,
        data: dict[str, Any],
        episode: AgentEpisodeResult | RewardBreakdown,
    ) -> None:
        if version <= 0 or version % self.audit_every_steps:
            return
        question = str(data.get("question", ""))
        trajectory_id = str(
            data.get("id")
            or data.get("_id")
            or hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
        )
        if isinstance(episode, RewardBreakdown):
            breakdown = episode
            process_rewards: dict[str, float] = {}
            transitions: tuple[CreditTransition, ...] = ()
        else:
            breakdown = episode.breakdown
            process_rewards = episode.process_rewards
            transitions = episode.transitions
        payload = {
            "step": version,
            "trajectory_id": trajectory_id,
            **asdict(breakdown),
            "credit_assignment": getattr(self, "credit_assignment", "terminal"),
            "process_reward_total": sum(process_rewards.values()),
            "nonzero_process_interactions": sum(
                reward != 0.0 for reward in process_rewards.values()
            ),
            "credit_transitions": [asdict(transition) for transition in transitions],
        }
        _append_audit_record(
            self.audit_path,
            payload,
            self.audit_sample_size,
        )

    async def arun_episode(self, engine: Any, data: dict[str, Any]) -> Any:
        from areal import workflow_context
        from areal.experimental.openai import ArealOpenAI
        from areal.utils import stats_tracker

        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            engine_max_tokens=self.max_total_tokens,
            chat_template_type="hf",
            lora_name=str(self.gconfig.lora_name),
        )
        context = workflow_context.get()
        episode = await self.agent.run(
            data=data,
            client=client,
            version=int(engine.get_version()),
            sample_idx=context.sample_idx,
        )
        breakdown = episode.breakdown
        reward = breakdown.total
        version = int(engine.get_version())
        await self._record_audit(version, data, episode)
        if self.credit_assignment == "abc":
            for interaction_id, process_reward in episode.process_rewards.items():
                client.set_reward(
                    interaction_id,
                    reward + self.abc_beta * process_reward,
                )
            interactions = client.export_interactions(style=self.export_style)
            for interaction_id, interaction in interactions.items():
                interaction.episode_reward = reward
                interaction.process_reward = episode.process_rewards.get(interaction_id, 0.0)
                interaction.process_reward_weight = self.abc_beta
        else:
            client.set_last_reward(reward)
            client.apply_reward_discount(turn_discount=self.turn_discount)
            interactions = client.export_interactions(style=self.export_style)
        process_values = list(episode.process_rewards.values())
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward,
            answer_f1=breakdown.answer_f1,
            support_recall=breakdown.support_recall,
            citation_precision=breakdown.citation_precision,
            duplicate_rate=breakdown.duplicate_rate,
            valid_reward=float(breakdown.valid),
            process_reward_total=sum(process_values),
            process_reward_nonzero_ratio=(
                sum(value != 0.0 for value in process_values) / len(process_values)
                if process_values
                else 0.0
            ),
        )
        return interactions
