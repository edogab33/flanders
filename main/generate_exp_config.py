import pandas as pd

def generate_d(
    pool_size=[10],
    fraction_fit=[1],
    fraction_evaluate=[0],
    malicious_clients=[0],
    min_fit_clients=[10],
    min_evaluate_clients=[0],
    magnitude=[0],
    warmup_rounds=[0],
    to_keep=[10],
    threshold=[1e-5],
    attack_name=["no attack"],
    strategy_name=["avg"],
    dataset_name=["circles"],
):
    d = {
        "pool_size": pool_size,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": fraction_evaluate,
        "malicious_clients": malicious_clients,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "magnitude": magnitude,
        "warmup_rounds": warmup_rounds,
        "to_keep": to_keep,
        "threshold": threshold,
        "attack_name": attack_name,
        "strategy_name": strategy_name,
        "dataset_name": dataset_name,
    }
    return d

def all_combinations():
    attacks = ["no attack", "gaussian", "lie", "fang", "minmax"]
    strategies = ["avg", "median", "trimmedmean", "krum", "multikrum", "fltrust", "flanders"]
    datasets = ["circles", "mnist", "cifar", "income"]
    #malicious_num = [0, 5, 10, 20, 30, 50]
    malicious_num = [0, 2, 4]

    d = {
        "pool_size": [],
        "fraction_fit": [],
        "fraction_evaluate": [],
        "malicious_clients": [],
        "min_fit_clients": [],
        "min_evaluate_clients": [],
        "magnitude": [],
        "warmup_rounds": [],
        "to_keep": [],
        "threshold": [],
        "attack_name": [],
        "strategy_name": [],
        "dataset_name": [],
    }

    for dataset in datasets:
        for strategy in strategies:
            for attack in attacks:
                for malicious in malicious_num:
                    d["pool_size"].append(10)
                    d["fraction_fit"].append(1)
                    d["fraction_evaluate"].append(0)
                    d["malicious_clients"].append(malicious)
                    d["min_fit_clients"].append(10)
                    d["min_evaluate_clients"].append(0)
                    if attack == "gaussian":
                        d["magnitude"].append(0.5)
                    else:
                        d["magnitude"].append(0)
                    if strategy == "flanders":
                        d["warmup_rounds"].append(70)
                    else:
                        d["warmup_rounds"].append(0)
                    d["to_keep"].append(10-malicious)
                    d["threshold"].append(1e-5)
                    d["attack_name"].append(attack)
                    d["strategy_name"].append(strategy)
                    d["dataset_name"].append(dataset)
    return d

d = generate_d(
    pool_size=[10,10],
    fraction_fit=[1,1],
    fraction_evaluate=[0,0],
    malicious_clients=[0,0],
    min_fit_clients=[10,10],
    min_evaluate_clients=[0,0],
    magnitude=[0,0],
    warmup_rounds=[1,10],
    to_keep=[10,6],
    threshold=[1e-5, 1e-5],
    attack_name=["no attack", "gaussian"],
    strategy_name=["avg", "flanders"],
    dataset_name=["income", "income"]
)
df = pd.DataFrame(data=d)
df.to_csv("experiments_config.csv", index=False)