import subprocess as sp, shlex, sys, datetime



SSH = ["ssh", f"{USER}@{HOST}"]

def run_ssh(label: str, cmd: str, timeout=25, tolerate_fail=False):
    print(f"\n[{label}]")
    print("$", " ".join(shlex.quote(x) for x in (SSH + [cmd])))
    res = sp.run(SSH + [cmd], capture_output=True, text=False, timeout=timeout)
    out = res.stdout.decode("utf-8", "replace").rstrip()
    err = res.stderr.decode("utf-8", "replace").rstrip()
    if out: print(out)
    if err: print(err)
    ok = (res.returncode == 0) or tolerate_fail
    print(f"-> exit: {res.returncode} ({'ok' if ok else 'fail'})")
    return ok

def main():
    print("== display Cubit via Scheduled Task in interactive session ==")

    if not run_ssh("echo", 'cmd /c echo OK'):
        sys.exit("SSH failed")

    from cubitpy.cubit_wrapper.cubit_wrapper_host import CubitConnect
    cc = CubitConnect(use_ssh=True)
    c = cc.cubit
    c.cmd("reset")
    c.cmd("create brick x 1 y 1 z 1")
    c.cmd("create block 1")
    c.cmd("block 1 volume 1")
    c.cmd("block 1 element type hex")
    c.cmd("mesh volume 1")
    c.cmd(fr'save cub5 "{STATE_CUB}" overwrite journal')
    print("blocks:", c.get_block_id_list())
    print("vols  :", c.get_entities("volume"))

    jou_cmd = (
        r'cmd /c ('
        fr'echo open "{STATE_CUB}"'
        r' & echo label volume On'
        r' & echo display'
        fr') > "{JOURNAL}"'
    )
    if not run_ssh("write journal", jou_cmd):
        sys.exit("Failed to write journal on Windows")

    # Sanity: EXE & JOU exist
    run_ssh("check EXE", fr'cmd /c if exist "{WIN_CUBIT_EXE}" (echo EXE OK) else echo EXE MISSING')
    run_ssh("check JOU", fr'cmd /c if exist "{JOURNAL}" (echo JOU OK) else echo JOU MISSING')

    # Delete old task 
    run_ssh("cleanup old task", fr'schtasks /Delete /TN {TASK_NAME} /F', tolerate_fail=True)

    st = (datetime.datetime.now() + datetime.timedelta(minutes=1)).strftime("%H:%M")
    create = (
        fr'schtasks /Create /TN {TASK_NAME} /SC ONCE /ST {st} /TR '
        fr'"\"{WIN_CUBIT_EXE}\" -nojournal -information Off -input \"{JOURNAL}\"" '
        r'/RL HIGHEST /IT /RU  /F'
    )
    if not run_ssh("create scheduled task", create):
        sys.exit("Task create failed")

    run_ssh("run scheduled task", fr'schtasks /Run /TN {TASK_NAME}', tolerate_fail=True)

    print("\n If a user is logged in on the Windows VM as 'USERNAME', Cubit should appear shortly.")
    print(f"   DB : {STATE_CUB}")
    print(f"   JOU: {JOURNAL}")
    print(f"   EXE: {WIN_CUBIT_EXE}")
    print(f"   Task: {TASK_NAME}")

if __name__ == "__main__":
    main()
