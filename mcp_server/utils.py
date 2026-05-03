def format_job_details(job: dict) -> str:
    fields = [
        ("Job ID",       job.get("job_id")),
        ("Name",         job.get("name")),
        ("User",         job.get("user_name")),
        ("State",        job.get("job_state")),
        ("Partition",    job.get("partition")),
        ("Nodes",        job.get("nodes")),
        ("CPUs",         job.get("cpus", {}).get("number")),
        ("Memory",       job.get("memory_per_node", {}).get("number")),
        ("Time limit",   job.get("time_limit", {}).get("number")),
        ("Submit time",  job.get("submit_time", {}).get("number")),
        ("Start time",   job.get("start_time", {}).get("number")),
        ("Working dir",  job.get("current_working_directory")),
        ("Stdout",       job.get("standard_output")),
        ("Stderr",       job.get("standard_error")),
        ("Exit code",    job.get("exit_code", {}).get("return_code")),
    ]
    lines = [f"**{k}**: {v}" for k, v in fields if v is not None]
    return "\n".join(lines)


def format_job_list(jobs: list[dict]) -> str:
    if not jobs:
        return "No jobs found."
    rows = ["| Job ID | Name | User | State | Partition | Time |"]
    rows.append("|--------|------|------|-------|-----------|------|")
    for j in jobs:
        rows.append(
            f"| {j.get('job_id','')} "
            f"| {j.get('name','')[:20]} "
            f"| {j.get('user_name','')} "
            f"| {j.get('job_state','')} "
            f"| {j.get('partition','')} "
            f"| {j.get('time_limit',{}).get('number','')} |"
        )
    return "\n".join(rows)
