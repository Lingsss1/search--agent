import unittest
from pathlib import Path

from openstatesearch.data.sources import huggingface_dataset_id, load_sources


ROOT = Path(__file__).resolve().parents[1]


class SourceTests(unittest.TestCase):
    def test_huggingface_dataset_url(self):
        self.assertEqual(
            huggingface_dataset_id("https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train"),
            "Yuqi-Zhou/LRAT-Train",
        )
        self.assertIsNone(huggingface_dataset_id("https://github.com/hotpotqa/hotpot"))

    def test_every_source_is_commit_pinned(self):
        sources = load_sources(ROOT / "data" / "manifests" / "sources.json")
        self.assertEqual(
            {source["name"] for source in sources},
            {
                "OpenSeeker-v1-Data",
                "OpenResearcher-Dataset",
                "LRAT-Train",
                "HotpotQA",
                "2WikiMultiHopQA",
                "MuSiQue-Ans",
                "BrowseComp-Plus",
                "xbench-DeepSearch",
                "BrowseComp-ZH",
            },
        )
        self.assertTrue(all(len(source["revision"]) == 40 for source in sources))


if __name__ == "__main__":
    unittest.main()
