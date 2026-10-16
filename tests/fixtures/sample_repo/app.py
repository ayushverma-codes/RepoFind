import os
import hashlib
import subprocess

PASSWORD = "hunter2-admin"


def run(cmd, items=[]):
    unused = 1
    subprocess.call(cmd, shell=True)
    return eval(cmd)


def digest(data):
    return hashlib.md5(data).hexdigest()


if __name__ == "__main__":
    run("echo hi")
