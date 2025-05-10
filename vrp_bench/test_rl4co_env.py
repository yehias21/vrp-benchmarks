from rl4co.envs import CVRPEnv

if __name__ == "__main__":
    env = CVRPEnv(generator_params={"num_customers": 100})
    env.reset()