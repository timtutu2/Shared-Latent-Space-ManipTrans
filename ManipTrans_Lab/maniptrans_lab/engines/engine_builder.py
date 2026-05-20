def build_engine(config, num_envs, device, visualize, record_video=False):
    eng_name = config["engine_name"]

    if (eng_name == "isaac_lab"):
        import engines.isaac_lab_engine as isaac_lab_engine
        engine = isaac_lab_engine.IsaacLabEngine(config, num_envs, device, visualize, record_video=record_video)
    else:
        assert False, "Unsupported engine: {:s}".format(eng_name)

    return engine
