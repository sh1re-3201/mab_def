import numpy as np
from tqdm import tqdm
from src.agents import SingleAgent, GossipingAgent, MaliciousAgent

def run_single_agent_baseline(means, stds, T, policy="UCB", explore_param=4.0, pos=0):
    K = len(means)
    best_mean = np.max(means)
    agent = SingleAgent(K, policy, explore_param)
    
    cumulative_regret = 0
    regret_history = []
    reward_history = []
    
    # mininterval=2.0 mencegah Terminal Windows menjadi macet (lag)
    for t in tqdm(range(T), desc=f"Single Agent ({policy})", position=pos, leave=True, mininterval=2.0):
        arm = agent.choose()
        
        # Fast Scalar Random Normal
        reward = np.random.normal(means[arm], stds[arm])
        reward = max(0.0, min(1.0, reward)) # Pengganti np.clip yang jauh lebih ringan
        
        agent.update(arm, reward)
        cumulative_regret += (best_mean - means[arm])
        regret_history.append(cumulative_regret)
        reward_history.append(reward)
        
    return np.array(regret_history), np.array(reward_history)


def run_multi_agent_system(
    means,
    stds,
    T,
    n_honest,
    m_malicious,
    beta,
    policy="UCB",
    explore_param=4.0,
    pos=0,
    defense="none",
    eta=2.0,
    lam=0.85,
    theta_min=-0.50,
):
    K = len(means)
    best_mean = np.max(means)
    best_arm_idx = np.argmax(means)
    # Total agents in the network (honest indices 0..n_honest-1, malicious indices n_honest..n_total-1).
    n_total = n_honest + m_malicious
    
    S_size = max(2, int((K / n_honest) * np.log(1/0.1)))
    all_arms = list(range(K))
    np.random.shuffle(all_arms)
    
    if best_arm_idx in all_arms: all_arms.remove(best_arm_idx)
    
    honest_agents = []
    for i in range(n_honest):
        if i == 0:
            sticky = [best_arm_idx] + all_arms[:S_size-1]
            all_arms = all_arms[S_size-1:]
        else:
            sticky = all_arms[:S_size]
            all_arms = all_arms[S_size:]
        honest_agents.append(
            GossipingAgent(
                K=K,
                sticky_set=sticky,
                beta=beta,
                policy=policy,
                param=explore_param,
                n_total=n_total,
                defense=defense,
                eta=eta,
                lam=lam,
                theta_min=theta_min,
            )
        )
        
    malicious_agents = [MaliciousAgent(K) for _ in range(m_malicious)]
    all_agents = honest_agents + malicious_agents
    
    cumulative_regret = 0
    regret_history = []
    reward_history = []
    
    desc_str = f"Multi-Agent ({defense}, n={n_honest}, m={m_malicious})"
    # mininterval=2.0 mencegah Terminal Windows menjadi macet (lag)
    for t in tqdm(range(1, T + 1), desc=desc_str, position=pos, leave=True, mininterval=2.0):
        step_regret = 0
        step_reward = 0
        
        for agent in honest_agents:
            arm = agent.choose()
            
            # Fast Scalar Random Normal
            reward = np.random.normal(means[arm], stds[arm])
            reward = max(0.0, min(1.0, reward))
            
            agent.update(arm, reward)
            step_regret += (best_mean - means[arm])
            step_reward += reward
            
        cumulative_regret += (step_regret / n_honest)
        regret_history.append(cumulative_regret)
        reward_history.append(step_reward / n_honest)
        
        # Communication Phase (2-Pass Synchronous Phase Boundary — ALGORITHMS.md Section 5.1)
        # All honest agents share the same phase schedule A_j = ceil(j^beta), so checking
        # honest_agents[0].check_end_phase() triggers synchronously for every honest agent.
        if honest_agents[0].check_end_phase():
            # PASS 1a: Snapshot every honest agent's true Phase j winner B_j^{(i)} BEFORE any
            # pulls_in_phase array is reset to zero by end_phase().
            phase_winners = [ag.get_recommendation_to_send() for ag in honest_agents]

            # PASS 1b: Audit the peer H_{j-1}^{(i)} that recommended R_{j-1}^{(i)} at the end
            # of Phase j-1 by comparing phase_winners[i] (B_j^{(i)}) against agent.last_rec.
            for i, agent in enumerate(honest_agents):
                if agent.j > 1 and agent.last_peer is not None:
                    agent.audit_previous_peer(phase_winners[i])

            # PASS 2: Filter candidate peers, select H_j^{(i)}, exchange recommendation, & reset phase.
            for i, agent in enumerate(honest_agents):
                h = agent.select_peer(agent_idx=i, n_total=n_total)

                if h is not None:
                    # Honest agents occupy indices 0..n_honest-1; malicious agents occupy n_honest..n_total-1.
                    if h < n_honest:
                        rec_arm = phase_winners[h]
                    else:
                        rec_arm = all_agents[h].get_recommendation_to_send()

                    # Unconditionally save H_j^{(i)} and R_j^{(i)} so Phase j+1 audits this query
                    # even if rec_arm was already present in agent.active_set.
                    agent.last_peer = h
                    agent.last_rec = rec_arm
                    agent.end_phase(rec_arm)
                else:
                    # No eligible peers exist (empty U_j^{(i)} or T_j^{(i)}): clear last-query
                    # memory so no false audit runs at Phase j+1, skip Insert-Eliminate, and advance phase.
                    agent.last_peer = None
                    agent.last_rec = None
                    agent.end_phase(None)
                
    return np.array(regret_history), np.array(reward_history)