"""hook 计数持久化地板。

跨帧稳定 raw hook 计数: 一次抬升必须连续 k 帧都观测到才采纳,
消除开局/转场的孤立单簇闪现误检(BV1pht hooks 恒 0 场景);
HUD 消失(hud_ok=False)帧不更新输出, 复用前一稳定值。
"""


class HookPersist:
    def __init__(self, k=4, n_players=4):
        self.k = k
        self.n = n_players
        self.reset()

    def reset(self):
        self._out = [0] * self.n
        self._cand = [0] * self.n
        self._cnt = [0] * self.n

    def update(self, raw, hud_ok=True):
        for i in range(self.n):
            v = raw[i]
            if v == self._out[i]:
                self._cand[i] = v
                self._cnt[i] = 0
                continue
            if hud_ok and v == self._cand[i]:
                self._cnt[i] += 1
            else:
                self._cand[i] = v
                self._cnt[i] = 1
            if hud_ok and self._cnt[i] >= self.k:
                self._out[i] = v
        return list(self._out)
