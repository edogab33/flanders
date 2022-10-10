import pandas as pd

def generate_d(
    window = [40],
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
    num_rounds=[50],
    sampling=[0],
):
    d = {
        "window":window,
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
        "num_rounds": num_rounds,
        "sampling": sampling,
    }
    return d

def all_combinations():
    attacks = ["minmax", "lie", "gaussian"]
    strategies = ["krum", "trimmedmean", "flanders"]
    datasets = ["mnist"]
    malicious_num = [35]
    to_keep = [50]

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
        "sampling": [],
    }

    for dataset in datasets:
        for strategy in strategies:
            for attack in attacks:
                for malicious in malicious_num:
                    for k in to_keep:
                        d["window"].append(10)
                        d["pool_size"].append(100)
                        d["fraction_fit"].append(1)
                        d["fraction_evaluate"].append(0)
                        d["malicious_clients"].append(malicious)
                        d["min_fit_clients"].append(100)
                        d["min_evaluate_clients"].append(0)
                        if attack == "gaussian":
                            d["magnitude"].append(0.5)
                        else:
                            d["magnitude"].append(0)
                        d["warmup_rounds"].append(10)
                        d["to_keep"].append(k)
                        d["threshold"].append(1e-5)
                        d["attack_name"].append(attack)
                        d["strategy_name"].append(strategy)
                        d["dataset_name"].append(dataset)
                        d["num_rounds"].append(20)
                        if strategy == "flanders" and (dataset == "cifar" or dataset == "mnist"):
                            d["sampling"].append(1000)
                        else:
                            d["sampling"].append(0)
    return d

d = all_combinations()
#d = generate_d(
#    window=[40,40,40],
#    pool_size=[5,10,10],
#    fraction_fit=[1,1,1],
#    fraction_evaluate=[0,0,0],
#    malicious_clients=[2,0,4],
#    min_fit_clients=[5,10,10],
#    min_evaluate_clients=[0,0,0],
#    magnitude=[0,0,0],
#    warmup_rounds=[10,10,60],
#    to_keep=[2,6,6],
#    threshold=[1e-5, 1e-5, 1e-5],
#    attack_name=["minmax", "no attack", "lie"],
#    strategy_name=["flanders", "avg", "flanders"],
#    dataset_name=["mnist", "income", "income"],
#    num_rounds=[50,50,50],
#    sampling=[50,0,0],
#)
df = pd.DataFrame(data=d)
df.to_csv("experiments_config.csv", index=False)