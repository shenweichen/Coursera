import sys
import numpy as np
sys.path.append("..")
import grading


def submit_cartpole(generate_session, email, token):
    sessions = [generate_session() for _ in range(100)]
    session_rewards = np.array(sessions)
    grader = grading.Grader("oyT3Bt7yEeeQvhJmhysb5g")
    grader.set_answer("7QKmA", int(np.mean(session_rewards)))
    grader.submit(email, token)


def submit_kungfu(agent, env, evaluate, email, token):
    sessions = [evaluate(agent, env, n_games=1) for _ in range(100)]
    session_rewards = np.array(sessions)
    grader = grading.Grader("6sPnVCn6EeieSRL7rCBNJA")
    grader.set_answer("HhNVX", 100*int(np.mean(session_rewards)))#scaling issue https://www.coursera.org/learn/practical-rl/discussions/weeks/5/threads/F1JHgGRmEei8iRK3RpIYPA/replies/vILx4WchEeimAgoMPemTPA
    grader.submit(email, token)
