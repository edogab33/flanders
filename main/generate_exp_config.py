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
    attacks = ["minmax"]
    strategies = ["avg", "median", "trimmedmean", "krum", "multikrum", "fltrust", "flanders"]
    datasets = ["income"]
    malicious_num = [0, 5, 10, 20, 30, 50]
    to_keep = [10, 25, 50]
    #malicious_num = [0, 2, 4]

    d = {
        "window":[],
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
        "num_rounds": [],
    }

    for dataset in datasets:
        for strategy in strategies:
            for attack in attacks:
                for malicious in malicious_num:
                    for k in to_keep:
                        d["window"].append(40)
                        d["pool_size"].append(100)
                        d["fraction_fit"].append(1)
                        d["fraction_evaluate"].append(0)
                        d["malicious_clients"].append(malicious)
                        d["min_fit_clients"].append(100)
                        d["min_evaluate_clients"].append(0)
                        if attack == "gaussian":
                            d["magnitude"].append(0.5)
                        d["magnitude"].append(0)
                        d["warmup_rounds"].append(40)
                        d["to_keep"].append(k)
                        d["threshold"].append(1e-5)
                        d["attack_name"].append(attack)
                        d["strategy_name"].append(strategy)
                        d["dataset_name"].append(dataset)
                        d["num_rounds"].append(50)
    return d

d = all_combinations()
#d = generate_d(
#    pool_size=[5,10,10],
#    fraction_fit=[1,1,1],
#    fraction_evaluate=[0,0,0],
#    malicious_clients=[0,0,4],
#    min_fit_clients=[10,10,10],
#    min_evaluate_clients=[0,0,0],
#    magnitude=[0,0,0],
#    warmup_rounds=[1,10,60],
#    to_keep=[10,6,6],
#    threshold=[1e-5, 1e-5, 1e-5],
#    attack_name=["no attack", "no attack", "lie"],
#    strategy_name=["avg", "avg", "flanders"],
#    dataset_name=["mnist", "income", "income"]
#)
df = pd.DataFrame(data=d)
df.to_csv("experiments_config.csv", index=False)