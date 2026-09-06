import unittest

import hook_persist


class TestHookPersist(unittest.TestCase):
    def test_single_flash_frame_is_rejected(self):
        """单簇闪现(孤立 1 帧 raw=1)不应让稳定输出抬升, BV1pht hooks 恒 0 场景防误检。"""
        h = hook_persist.HookPersist(k=4)
        seq = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for raw in seq:
            out = h.update(raw)
        self.assertEqual(out, [0, 0, 0, 0])

    def test_sustained_hook_raises_after_k_frames(self):
        h = hook_persist.HookPersist(k=4)
        for _ in range(3):
            out = h.update([0, 1, 0, 0])
        self.assertEqual(out, [0, 0, 0, 0])
        out = h.update([0, 1, 0, 0])
        self.assertEqual(out, [0, 1, 0, 0])

    def test_release_needs_sustained_zero(self):
        """稳定输出为 1 后, 单帧 0(短暂丢失)不应把输出拉回 0。"""
        h = hook_persist.HookPersist(k=4)
        for _ in range(4):
            h.update([1, 0, 0, 0])
        out = h.update([0, 0, 0, 0])
        self.assertEqual(out, [1, 0, 0, 0])
        out = h.update([0, 0, 0, 0])
        self.assertEqual(out, [1, 0, 0, 0])
        for _ in range(3):
            out = h.update([0, 0, 0, 0])
        self.assertEqual(out, [0, 0, 0, 0])

    def test_hud_gone_reuses_previous_value(self):
        """HUD 消失(结算/转场)帧 hooks 不可信, 输出应复用前值而非误报 0。"""
        h = hook_persist.HookPersist(k=4)
        for _ in range(4):
            h.update([1, 0, 0, 0])
        for _ in range(10):
            out = h.update([0, 0, 0, 0], hud_ok=False)
            self.assertEqual(out, [1, 0, 0, 0])

    def test_players_independent(self):
        h = hook_persist.HookPersist(k=2)
        h.update([1, 0, 0, 0])
        h.update([1, 0, 0, 0])
        out = h.update([1, 0, 0, 0])
        self.assertEqual(out[0], 1)
        self.assertEqual(out[1:], [0, 0, 0])

    def test_reset_clears_state(self):
        h = hook_persist.HookPersist(k=2)
        for _ in range(2):
            h.update([1, 0, 0, 0])
        h.reset()
        out = h.update([0, 0, 0, 0])
        self.assertEqual(out, [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
