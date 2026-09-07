import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import prescan

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BV1U = os.path.join(BASE, "picture", "BV1Uu8z6eEVM")
BV16 = os.path.join(BASE, "picture", "BV16QtT6ZEPq")


def _samples(gens, sims):
    """构造 prescan.Sample 序列。sims[i]=sim_prev(与最近前一含头像样本的 NCC 均值)。"""
    out = []
    for i, g in enumerate(gens):
        out.append(prescan.Sample(
            t=float(i), fname=f"frame_{i}.jpg", gens=g,
            portraits=None if sims is None else [[0]] * 4,
            sim_prev=sims[i] if sims is not None else None,
            anchor=None))
    return out


class TestFindBoundaries(unittest.TestCase):
    def test_monotonic_decreasing_no_boundary(self):
        gens = [5, 5, 4, 3, 2, 1, 0, 0]
        self.assertEqual(prescan.find_boundaries(_samples(gens, None)), [])

    def test_gens_jump_with_changed_portraits_is_boundary(self):
        gens = [5, 4, 3, 2, 0, 0, 5]
        sims = [None, 1.0, 1.0, 1.0, 1.0, 1.0, 0.30]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [6])

    def test_gens_jump_but_same_portraits_no_boundary(self):
        """BV1aat 类：gens 异常跳回 5 但头像(同一批幸存者)未变 -> 不切局。"""
        gens = [5, 4, 3, 2, 0, 0, 5]
        sims = [None, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [])

    def test_gap_then_same_gens_needs_portrait_change(self):
        """末局在 5 时被速杀 -> 换局前后 gens 都是 5: 需靠头像差异判定。"""
        gens = [5, 5, 5, None, None, 5]
        changed = [None, 1.0, 1.0, None, None, 0.25]
        same = [None, 1.0, 1.0, None, None, 0.9]
        self.assertEqual(prescan.find_boundaries(_samples(gens, changed)), [5])
        self.assertEqual(prescan.find_boundaries(_samples(gens, same)), [])

    def test_jump_without_portrait_compare_is_conservative(self):
        """跳变成立但前后无可靠头像可比(如 HUD 整段消失) -> 不切, 交给全量兜底。"""
        gens = [5, 4, 2, 0, 5]
        sims = [None, None, None, None, None]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [])

    def test_momentary_gap_same_gens_not_boundary(self):
        """BV1Uu 型: 单样本瞬间不可读(None)后同值复现, 头像 NCC 偶发略低 -> 不是切局。"""
        gens = [5, 4, 2, None, 2]
        sims = [None, 1.0, 1.0, None, 0.2]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [])

    def test_sustained_gap_same_gens_is_boundary(self):
        """持续不可读(>=2 样本)或识别到 0 后同值复现 + 头像明显不同 -> 真切局。"""
        gens = [5, None, None, 5]
        sims = [None, None, None, 0.1]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [3])

    def test_noise_dip_recovery_not_boundary(self):
        """单帧噪声读低(4->3 误读)后回真值 4 是"上升", 但头像没变 -> 不切。"""
        gens = [5, 4, 3, 4, 4, 4]
        sims = [None, 1.0, 1.0, 0.98, 1.0, 1.0]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [])

    def test_two_real_boundaries(self):
        gens = [5, 4, 3, 2, 0, 0, 5, 5, 4, 2, 1, 0, 5]
        sims = [None] + [1.0] * 5 + [0.2] + [1.0] * 5 + [0.3]
        self.assertEqual(prescan.find_boundaries(_samples(gens, sims)), [6, 12])


class TestAvatarBoundary(unittest.TestCase):
    """头像区骤变 = 旧局结束(换人/结算)。BV1QUt766Etg 型: gens 修完后残局
    长时间 0, gens 不再回 5, 旧 gens 切局规则失效/滞后; 应以头像内容骤变切局。"""

    def _frames(self, n):
        return [[0]] * 4

    def test_sustained_avatar_change_is_boundary(self):
        """稳定 HUD 后连续 >=2 样本头像骤变(sim 低) -> 切在首次骤变处(旧局结束)。"""
        gens = [5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0]
        sims = [None] + [0.9, 0.9, 0.85, 0.8, 0.7, 0.1, 0.05, 0.9, 0.8, 0.85]
        samples = []
        for i, g in enumerate(gens):
            samples.append(prescan.Sample(
                t=float(i), fname=f"f{i}", gens=g, portraits=self._frames(1),
                sim_prev=sims[i]))
        got = prescan.find_boundaries(samples)
        self.assertEqual(got, [6])

    def test_momentary_avatar_dip_is_not_boundary(self):
        """单样本 sim 跌落(状态变化)后回稳 -> 同局, 不切。"""
        gens = [5, 4, 3, 3, 3, 3, 3]
        sims = [None, 0.9, 0.8, 0.2, 0.9, 0.85, 0.8]
        samples = []
        for i, g in enumerate(gens):
            samples.append(prescan.Sample(
                t=float(i), fname=f"f{i}", gens=g, portraits=self._frames(1),
                sim_prev=sims[i]))
        self.assertEqual(prescan.find_boundaries(samples), [])


class TestConsensusCandidates(unittest.TestCase):
    """跨帧全候选共识: 单帧最高分被常驻高分伪匹配(位置乱跳)压过时,
    真实 HUD 图标(位置稳定、支持帧数最多)仍应被选为共识锚点。"""

    def _real(self, score=0.75):
        return {"scale": 1.6, "score": score, "box": (213, 804, 269, 855)}

    def _moving_false(self, y, score=0.95):
        return {"scale": 1.1, "score": score, "box": (10, y, 48, y + 35)}

    def test_stable_real_beats_high_score_moving_false(self):
        cands_per = []
        moving_ys = [18, 97, 178, 218, 57]
        for i in range(8):
            row = [self._real()]
            if i >= 3:
                row.append(self._moving_false(moving_ys[i - 3]))
            cands_per.append(row)
        anchor, ratio = prescan._consensus_candidates(cands_per)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], 213, delta=5)
        self.assertAlmostEqual(anchor["y"], 804, delta=5)
        self.assertAlmostEqual(anchor["scale"], 1.6, delta=0.1)
        self.assertGreaterEqual(ratio, 0.8)

    def test_second_layout_cluster_not_confused(self):
        """另一段 HUD 布局(A 换 B)固定位置不同 -> 取支持帧数更多者。"""
        b = {"scale": 1.2, "score": 0.9, "box": (900, 400, 942, 442)}
        cands_per = [[self._real()] for _ in range(6)] + [[dict(b)] for _ in range(3)]
        anchor, ratio = prescan._consensus_candidates(cands_per)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor["x"], 213, delta=5)
        self.assertGreaterEqual(ratio, 0.6)


class TestPrescanReal(unittest.TestCase):
    """真实帧(10s 抽样目录)端到端: HUD 锚点确认 + 多局判定。"""

    def test_BV1Uu_anchors_and_boundary(self):
        res = prescan.run_prescan(BV1U, max_frames=68)
        self.assertIsNotNone(res.anchor, "BV1Uu 应能确认 HUD 锚点")
        self.assertGreaterEqual(res.anchor_ratio, 0.4)
        self.assertAlmostEqual(res.anchor["x"], 121, delta=20)
        self.assertAlmostEqual(res.anchor["y"], 847, delta=20)
        self.assertEqual(len(res.boundaries), 1)
        b = res.samples[res.boundaries[0]]
        self.assertGreaterEqual(b.t, 600)
        self.assertLessEqual(b.t, 700)
        self.assertEqual(b.gens, 5)

    def test_BV16_single_match_no_boundary(self):
        res = prescan.run_prescan(BV16)
        self.assertIsNotNone(res.anchor)
        self.assertEqual(res.boundaries, [])


if __name__ == "__main__":
    unittest.main()
