def run_algorithm(algo_name, data):
    """
    Dispatch to the appropriate algorithm function.
    For now, return a dummy result.
    """
    if algo_name == "MST":
        return {"result": "MST algorithm executed"}
    return {"error": "Algorithm not implemented"}
