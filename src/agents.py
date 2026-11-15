import numpy as np

class SingleAgent:
    def __init__(self, K, policy="UCB", param=4.0):
        self.K = K
        self.policy = policy
        self.param = param
        self.pulls = np.zeros(K)
        self.rewards = np.zeros(K)
        self.t = 0

    def choose(self):
        self.t += 1
        unplayed = np.where(self.pulls == 0)[0]
        if len(unplayed) > 0:
            return unplayed[0]
            
        if self.policy == "UCB":
            indices = (self.rewards / self.pulls) + np.sqrt((self.param * np.log(self.t)) / self.pulls)
            return np.argmax(indices)
        elif self.policy == "Epsilon-Greedy":
            eps = min(1.0, self.param / self.t)
            if np.random.rand() < eps:
                return np.random.randint(self.K) 
            else:
                return np.argmax(self.rewards / self.pulls) 
        elif self.policy == "Thompson-Sampling":
            samples = np.random.beta(self.rewards + 1, (self.pulls - self.rewards) + 1)
            return np.argmax(samples)

    def update(self, arm, reward):
        self.pulls[arm] += 1
        self.rewards[arm] += reward

class MaliciousAgent:
    def __init__(self, K):
        self.K = K
    def get_recommendation_to_send(self):
        return np.random.randint(self.K)

class GossipingAgent:
    def __init__(
        self,
        K,
        sticky_set,
        beta,
        policy="UCB",
        param=4.0,
        n_total=25,
        defense="none",
        eta=2.0,
        lam=0.85,
        theta_min=-0.50,
    ):
        self.K = K
        self.sticky_set = list(sticky_set)
        self.beta = beta
        self.policy = policy
        self.param = param
        self.pulls = np.zeros(K)
        self.rewards = np.zeros(K)
        self.pulls_in_phase = np.zeros(K)
        self.t = 0
        self.j = 1
        self.A_j = int(np.ceil(self.j ** self.beta))

        # --- Defense Configuration Parameters (ALGORITHMS.md) ---
        # Total number of agents in the network (honest + malicious), used to size per-peer defense arrays.
        self.n_total = n_total
        # Active gossip defense algorithm: "none", "gosine_block", "pure_brs", or "hybrid_brs_block".
        self.defense = defense
        # Base polynomial blocking exponent (eta > 1) used in GosInE-Block (Formula 1.2) and Hybrid BRS-Block (Formula 4.1).
        self.eta = eta
        # Forgetting factor (lambda in (0, 1]) used in Beta Reputation updates (Formula 2.1) to decay older evidence.
        self.lam = lam
        # Minimum reputation threshold (theta_min in (-1, 0)) below which peers are quarantined in Pure BRS-GosInE (Formula 2.3).
        self.theta_min = theta_min

        # --- Per-Agent Gossip Audit & Trust State (ALGORITHMS.md Sections 2–4) ---
        # Peer index H_{j-1}^{(i)} queried at the end of the previous phase (None at Phase 1 or if no eligible peer existed).
        self.last_peer = None
        # Movie arm ID R_{j-1}^{(i)} recommended by H_{j-1}^{(i)} at the end of the previous phase.
        self.last_rec = None
        # Ban expiration phase d^{(i)}(h) for each peer h; peer h is blocked during phase j if blocked_until[h] >= j.
        self.blocked_until = np.zeros(n_total, dtype=np.int64)
        # Positive Beta reputation evidence r_{h,j}^{(i)} (recency-weighted count of winning recommendations from peer h).
        self.r_rep = np.zeros(n_total, dtype=float)
        # Negative Beta reputation evidence s_{h,j}^{(i)} (recency-weighted count of losing recommendations from peer h).
        self.s_rep = np.zeros(n_total, dtype=float)
        
        available = list(set(range(K)) - set(self.sticky_set))
        np.random.shuffle(available)
        self.U = available[0]
        self.L = available[1]
        self.active_set = self.sticky_set + [self.U, self.L]
        # OPTIMASI: SIMPAN SEBAGAI NUMPY ARRAY PERMANEN
        self.active_arms = np.array(self.active_set) 

    def choose(self):
        self.t += 1
        p = self.pulls[self.active_arms]
        r = self.rewards[self.active_arms]
        
        unplayed_idx = np.where(p == 0)[0]
        if len(unplayed_idx) > 0:
            return self.active_arms[unplayed_idx[0]]
            
        if self.policy == "UCB":
            indices = (r / p) + np.sqrt((self.param * np.log(self.t)) / p)
            return self.active_arms[np.argmax(indices)]
        elif self.policy == "Epsilon-Greedy":
            eps = min(1.0, self.param / self.t)
            if np.random.rand() < eps:
                return self.active_arms[np.random.randint(len(self.active_arms))]
            else:
                return self.active_arms[np.argmax(r / p)]
        elif self.policy == "Thompson-Sampling":
            samples = np.random.beta(r + 1, (p - r) + 1)
            return self.active_arms[np.argmax(samples)]

    def update(self, arm, reward):
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.pulls_in_phase[arm] += 1

    def check_end_phase(self):
        return self.t == self.A_j

    def get_recommendation_to_send(self):
        # Formula 1.1 (Phase Winner B_j^{(i)}): return the active arm with the most pulls in the current phase.
        p_phase = self.pulls_in_phase[self.active_arms]
        return self.active_arms[np.argmax(p_phase)]

    def get_expected_trust(self, h):
        """
        Compute expected trust E_{h,j}^{(i)} in (0, 1) for peer h (ALGORITHMS.md Formula 2.2).

        Uses the mean of Beta(r_{h,j}^{(i)} + 1, s_{h,j}^{(i)} + 1) with a uniform Beta(1, 1)
        prior so a brand-new peer (r = 0, s = 0) starts at a neutral 0.50 trust score.
        """
        return (self.r_rep[h] + 1.0) / (self.r_rep[h] + self.s_rep[h] + 2.0)

    def get_rep(self, h):
        """
        Compute normalized reputation rating Rep_{h,j}^{(i)} in (-1, +1) for peer h (ALGORITHMS.md Formula 2.2).

        Maps positive vs. negative evidence balance to (-1, +1) via (r - s) / (r + s + 2),
        where a brand-new peer (r = 0, s = 0) starts at a neutral 0.00 reputation.
        """
        return (self.r_rep[h] - self.s_rep[h]) / (self.r_rep[h] + self.s_rep[h] + 2.0)

    def audit_previous_peer(self, phase_winner):
        """
        Step 2 of the phase-end protocol (ALGORITHMS.md Section 5.1):
        Audit the peer H_{j-1}^{(i)} queried at the end of phase j-1 by checking whether
        their recommended arm R_{j-1}^{(i)} became our most-pulled arm B_j^{(i)} (`phase_winner`).
        """
        # Phase 1 has no prior recommendation to audit; likewise skip if no peer was queried last phase.
        if self.j <= 1 or self.last_peer is None:
            return

        # Algorithm 0 (Unprotected GosInE): performs no audit, blocking, or reputation tracking.
        if self.defense == "none":
            return

        # Index h of the peer H_{j-1}^{(i)} whose phase j-1 recommendation is now being judged.
        h = self.last_peer

        if self.defense == "gosine_block":
            # Algorithm 1 (GosInE-Block, Formula 1.2): 1-phase probation audit.
            # If the recommended arm failed to win phase j, block peer h from phase j through ceil(j^eta).
            if phase_winner != self.last_rec:
                self.blocked_until[h] = int(np.ceil(self.j ** self.eta))

        elif self.defense == "pure_brs":
            # Algorithm 2 (Pure BRS-GosInE, Formula 2.1): binary success indicator I_succ in {0, 1}.
            # Decay ONLY peer h's prior evidence by lambda and add the new outcome (unqueried peers stay untouched).
            i_succ = 1.0 if (phase_winner == self.last_rec) else 0.0
            self.r_rep[h] = (self.lam * self.r_rep[h]) + i_succ
            self.s_rep[h] = (self.lam * self.s_rep[h]) + (1.0 - i_succ)

        elif self.defense == "hybrid_brs_block":
            # Algorithm 3 (Hybrid BRS-Block, Formulas 2.1, 2.2, & 4.1):
            # Step A: Update peer h's Beta evidence (r, s) first using forgetting factor lambda.
            i_succ = 1.0 if (phase_winner == self.last_rec) else 0.0
            self.r_rep[h] = (self.lam * self.r_rep[h]) + i_succ
            self.s_rep[h] = (self.lam * self.s_rep[h]) + (1.0 - i_succ)

            # Step B: If the recommendation failed (I_succ == 0), dynamically scale Vial's blocking
            # exponent using peer h's updated reputation Rep_{h,j}^{(i)}.
            if i_succ == 0.0:
                rep_h = self.get_rep(h)
                # Formula 4.1: Positive reputation shrinks eta_tilde < eta (forgiving timeout);
                # negative reputation amplifies eta_tilde > eta (faster permanent lockout).
                eta_tilde = 1.0 + (self.eta - 1.0) * (1.0 - rep_h)
                # Cap at 1e9 solely to avoid 64-bit integer overflow when j ** eta_tilde is huge
                # (any expiration > J = 1,000 is already permanent over T = 1,000,000).
                ban_until = int(np.ceil(min(1e9, float(self.j) ** eta_tilde)))
                self.blocked_until[h] = ban_until

    def select_peer(self, agent_idx, n_total):
        """
        Step 3 of the phase-end protocol (ALGORITHMS.md Section 5.1):
        Build the eligible peer set (excluding self `agent_idx`) and sample one peer index
        H_j^{(i)} to query at the end of phase j, or return None if no peer is eligible.
        """
        if self.defense == "none":
            # Algorithm 0: All N - 1 other peers are eligible and sampled uniformly at random.
            eligible = [h for h in range(n_total) if h != agent_idx]
            if len(eligible) == 0:
                return None
            return int(np.random.choice(eligible))

        elif self.defense == "gosine_block":
            # Algorithm 1 (Formula 1.3): Keep only unblocked peers whose ban expired before phase j (d(h) < j).
            eligible = [h for h in range(n_total) if h != agent_idx and self.blocked_until[h] < self.j]
            if len(eligible) == 0:
                return None
            # Sample uniformly at random from the unblocked peer set U_j^{(i)}.
            return int(np.random.choice(eligible))

        elif self.defense == "pure_brs":
            # Algorithm 2 (Formula 2.3): Keep only unquarantined peers with Rep_{h,j}^{(i)} >= theta_min.
            eligible = [h for h in range(n_total) if h != agent_idx and self.get_rep(h) >= self.theta_min]
            if len(eligible) == 0:
                return None
            # Algorithm 2 (Formula 2.4): Sample peers proportionally to their expected trust E_{h,j}^{(i)}.
            weights = np.array([self.get_expected_trust(h) for h in eligible], dtype=float)
            probs = weights / np.sum(weights)
            return int(np.random.choice(eligible, p=probs))

        elif self.defense == "hybrid_brs_block":
            # Algorithm 3 (Step 3): Filter using hard escalating blocklist d(h) < j (no theta_min cutoff),
            # then sample among unblocked peers proportionally to expected trust E_{h,j}^{(i)} (Formula 2.4).
            eligible = [h for h in range(n_total) if h != agent_idx and self.blocked_until[h] < self.j]
            if len(eligible) == 0:
                return None
            weights = np.array([self.get_expected_trust(h) for h in eligible], dtype=float)
            probs = weights / np.sum(weights)
            return int(np.random.choice(eligible, p=probs))

    def end_phase(self, received_arm=None):
        """
        Step 4 of the phase-end protocol (ALGORITHMS.md Sections 1.3 & 5.1):
        If a peer was queried (`received_arm` is not None) and its recommended arm is outside
        our active set, run Insert-Eliminate: keep the more-pulled of {U, L} in U and put
        `received_arm` into L. Then reset phase pull counters and advance to phase j + 1.
        """
        if received_arm is not None and received_arm not in self.active_set:
            non_sticky = np.array([self.U, self.L])
            non_sticky_pulls = self.pulls_in_phase[non_sticky]
            self.U = non_sticky[np.argmax(non_sticky_pulls)]
            self.L = received_arm
            self.active_set = self.sticky_set + [self.U, self.L]
            # UPDATE NUMPY ARRAY HANYA SAAT FASE BERAKHIR
            self.active_arms = np.array(self.active_set) 
            
        self.pulls_in_phase.fill(0)
        self.j += 1
        self.A_j = int(np.ceil(self.j ** self.beta))