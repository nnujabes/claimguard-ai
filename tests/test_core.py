import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.fusion import fuse
from backend.decision import decide,optimize_roi
from backend.repository import ResultRepository

class CoreTests(unittest.TestCase):
    def test_fusion(self):
        r=fuse({"trufor":.8,"fused":.9,"recapture":.1});self.assertAlmostEqual(r.score,.745)
        self.assertEqual(r.dominant_model,"fused")
    def test_decision_boundary(self): self.assertTrue(decide(.63,.63).review);self.assertFalse(decide(.629,.63).review)
    def test_roi(self): self.assertIn("threshold",optimize_roi([0,1],[.1,.9],1000,10,2))
    def test_repository_roundtrip(self):
        r=ResultRepository(ROOT/"data"/"results");path=r.save("__UNIT_TEST__",{"final_score":.8})
        try: self.assertEqual(r.get("__UNIT_TEST__")["analyses"][0]["final_score"],.8)
        finally: path.unlink(missing_ok=True)
if __name__=="__main__":unittest.main()
