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
    # Generate all combinations of the parameters with lists of same length
    attacks = ["no attack", "gaussian", "lie", "fang", "minmax"]
    strategies = ["flanders"]
    datasets = ["cifar"]
    malicious_num = [20]
    to_keep = [1, 5, 20]
    warmup_rounds = 30
    magnitude = 10
    threshold = 1e-5
    sampling = 500
    window = 30
    alpha = 1
    beta = 1
    num_rounds = 50
    pool_size = 100
    fraction_fit = 1
    fraction_evaluate = 0
    min_fit_clients = 100
    min_evaluate_clients = 0

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
        "alpha": [],
        "beta": [],
    }

    for dataset in datasets:
        for strategy in strategies:
            for attack in attacks:
                for malicious in malicious_num:
                    for i, k in enumerate(to_keep):
                        if strategy not in ['multikrum', 'flanders', 'trimmedmean'] and i > 0:
                            continue
                        if (k + malicious) > pool_size:
                            continue
                        d["window"].append(window)
                        d["pool_size"].append(pool_size)
                        d["fraction_fit"].append(fraction_fit)
                        d["fraction_evaluate"].append(fraction_evaluate)
                        d["malicious_clients"].append(malicious)
                        d["min_fit_clients"].append(min_fit_clients)
                        d["min_evaluate_clients"].append(min_evaluate_clients)
                        if attack == "gaussian":
                            d["magnitude"].append(magnitude)
                        else:
                            d["magnitude"].append(0)
                        d["warmup_rounds"].append(warmup_rounds)
                        d["to_keep"].append(k)
                        d["threshold"].append(threshold)
                        d["attack_name"].append(attack)
                        d["strategy_name"].append(strategy)
                        d["dataset_name"].append(dataset)
                        d["num_rounds"].append(num_rounds)
                        if strategy == "flanders" and (dataset == "cifar" or dataset == "mnist"):
                            d["sampling"].append(sampling)
                        else:
                            d["sampling"].append(0)
                        if strategy == "flanders":
                            d["alpha"].append(alpha)
                            d["beta"].append(beta)
                        else:
                            d["alpha"].append(0)
                            d["beta"].append(0)
    print(d)
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