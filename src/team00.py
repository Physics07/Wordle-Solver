import atexit
import datetime
import json
import os

from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.cortex import complete

load_dotenv()


class Solver:
    def __init__(self):
        self.session = self._init_snowflake()
        self.model = "claude-3-5-sonnet"
        self.problems = {}
        self.snowflake_calls = 0
        self.log_file = open("run.log", "a")
        atexit.register(self.cleanup)

    def _init_snowflake(self):
        connection_params = {
            "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "user": os.environ.get("SNOWFLAKE_USER"),
            "password": os.environ.get("SNOWFLAKE_USER_PASSWORD"),
            "role": os.environ.get("SNOWFLAKE_ROLE"),
            "database": os.environ.get("SNOWFLAKE_DATABASE", ""),
            "schema": os.environ.get("SNOWFLAKE_SCHEMA", ""),
            "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
        }
        return Session.builder.configs(connection_params).create()

    def cleanup(self):
        try:
            self.log_file.close()
        except:
            pass
        try:
            self.session.close()
        except:
            pass

    def start_problem(self, problem_id, candidate_words):
        self.problems[problem_id] = {
            "candidate_words": candidate_words,
            "feedback_history": [],
        }
        self._log(f"\n=== Starting Problem {problem_id} ===")
        self._log(f"Candidate words: {', '.join(candidate_words)}")

    def add_feedback(self, problem_id, verbal_feedback):
        if verbal_feedback:
            self.problems[problem_id]["feedback_history"].append(verbal_feedback)

    def choose_next_guess(self, problem_id, turn):
        candidates = self.problems[problem_id]["candidate_words"]
        history = self.problems[problem_id]["feedback_history"]

        if not history:
            guess = candidates[0]
            self._log(f"Turn {turn}: Received feedback: None (first turn)")
            self._log(f"Turn {turn}: Guess: {guess}")
            return guess

        prompt_history = "\n".join([f"Feedback: {fb}" for fb in history])

        prompt = (
            "You're playing Wordle. Based on the following feedback, choose your next guess from the candidate list.\n"
            f"Feedback so far:\n{prompt_history}\n\n"
            f"Candidate words: {', '.join(candidates)}\n"
            "Return only your next guess."
        )

        response = (
            complete(
                model=self.model,
                prompt=[{"role": "user", "content": prompt}],
                options={"max_tokens": 10, "temperature": 0.0},
                session=self.session,
            )
            .strip()
            .lower()
        )

        self.snowflake_calls += 1

        guess = response if response in candidates else candidates[0]
        last_feedback = history[-1] if history else "None"
        self._log(f"Turn {turn}: Received feedback: {last_feedback}")
        self._log(f"Turn {turn}: Guess: {guess}")
        return guess

    def _log(self, msg):
        ts = datetime.datetime.now().isoformat()
        self.log_file.write(f"[{ts}] {msg}\n")
        self.log_file.flush()


solver = Solver()


class StudentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length"))
        data = json.loads(self.rfile.read(length))

        if self.path == "/start_problem":
            problem_id = data["problem_id"]
            candidate_words = data["candidate_words"]
            solver.start_problem(problem_id, candidate_words)
            self.send_response(200)
            self.end_headers()
            return

        if self.path == "/guess":
            problem_id = data["problem_id"]
            verbal_feedback = data.get("verbal_feedback")
            turn = data["turn"]
            solver.add_feedback(problem_id, verbal_feedback)
            guess = solver.choose_next_guess(problem_id, turn)

            response = {"guess": guess}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return

        self.send_response(404)
        self.end_headers()


def run():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), StudentHandler)
    print(f"Student server running on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    run()
