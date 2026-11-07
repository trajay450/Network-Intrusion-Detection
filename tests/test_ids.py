from network_ids.data import generate_flows
from network_ids.benchmark import run

def test_reproducible():
    assert generate_flows(50,3).equals(generate_flows(50,3))

def test_columns_labels():
    df=generate_flows(100)
    assert len(df)==100
    assert set(df.label.unique()) <= {0,1}
    assert not df.isna().any().any()

def test_metrics():
    result=run(seed=7,n=100,save=False)
    assert set(result)=={'logistic_regression','random_forest'}
    assert all(0<=v<=1 for d in result.values() for v in d.values())
