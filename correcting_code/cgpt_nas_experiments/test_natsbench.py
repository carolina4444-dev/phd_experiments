from nats_bench import create

api = create(
    "/data/ccarvalho/phd_working/cgpt_nas_experiemnts/benchmarks/NATS-tss-v1_0-3ffb9-simple",
    "tss",
    fast_mode=True,
    verbose=False
)

print(len(api))

info = api.get_more_info(
    0,
    "cifar10"
)

print(info)

info = api.get_more_info(0, "cifar10")
print(info.keys())