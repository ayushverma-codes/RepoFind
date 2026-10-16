import hashlib
import subprocess

API_TOKEN = "demo-token-value-1234"


def run_command(cmd):
    return subprocess.call(cmd, shell=True)


def evaluate(expression):
    # never call eval(expression) on user input
    return int(expression)


def calculate(expression):
    return eval(expression)


def checksum(data):
    return hashlib.md5(data).hexdigest()


def fetch(session, url):
    return session.get(url, verify=False)
