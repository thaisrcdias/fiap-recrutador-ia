import pandas as pd
from src.data_loader import DataLoader

def test_load_data_mock(monkeypatch):
    class FakeClient:
        def query(self, query):
            class FakeJob:
                def to_dataframe(self):
                    return pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
            return FakeJob()

    monkeypatch.setattr("src.data_loader.bigquery.Client", lambda project: FakeClient())

    loader = DataLoader("fake_project", "fake_view")
    df = loader.load_data()
    assert not df.empty
    assert "col1" in df.columns