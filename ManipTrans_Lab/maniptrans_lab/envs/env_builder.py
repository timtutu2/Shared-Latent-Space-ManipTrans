import yaml

# ensure engine builder is importable before env (simulator init ordering)
import engines.engine_builder as engine_builder  # noqa: F401

from util.logger import Logger


def build_env(env_file, engine_file, num_envs, device, visualize, record_video=False):
    env_config, engine_config = load_configs(env_file, engine_file)

    env_name = env_config["env_name"]
    Logger.print("Building {} env".format(env_name))

    if (env_name == "dex_imitator"):
        import envs.dex_imitator_env as dex_imitator_env
        env = dex_imitator_env.DexImitatorEnv(
            env_config=env_config, engine_config=engine_config,
            num_envs=num_envs, device=device,
            visualize=visualize, record_video=record_video)
    elif (env_name == "dex_manip_sh"):
        import envs.dex_manip_sh_env as dex_manip_sh_env
        env = dex_manip_sh_env.DexManipSHEnv(
            env_config=env_config, engine_config=engine_config,
            num_envs=num_envs, device=device,
            visualize=visualize, record_video=record_video)
    elif (env_name == "dex_manip_bih"):
        import envs.dex_manip_bih_env as dex_manip_bih_env
        env = dex_manip_bih_env.DexManipBiHEnv(
            env_config=env_config, engine_config=engine_config,
            num_envs=num_envs, device=device,
            visualize=visualize, record_video=record_video)
    else:
        assert False, "Unsupported env: {}".format(env_name)

    return env


def load_config(file):
    if file is not None and file != "":
        with open(file, "r") as stream:
            config = yaml.safe_load(stream)
    else:
        config = None
    return config


def load_configs(env_file, engine_file):
    env_config = load_config(env_file)
    engine_config = load_config(engine_file)

    if "engine" in env_config:
        env_engine_config = env_config["engine"]
        engine_config = override_engine_config(env_engine_config, engine_config)

    return env_config, engine_config


def override_engine_config(env_engine_config, engine_config):
    Logger.print("Overriding Engine configs with parameters from the Environment:")
    if engine_config is None:
        engine_config = env_engine_config
    else:
        engine_config = engine_config.copy()
        for key, val in env_engine_config.items():
            engine_config[key] = val
            Logger.print("\t{}: {}".format(key, val))
    Logger.print("")
    return engine_config
