import pickle
import math
import json

import pandas as pd
import numpy as np

    

def transform_transcript(portfolio_df, filepath="data/transcript.json"):
    # read in the json files
    df = pd.read_json(filepath, orient='records', lines=True)

    # extract value
    df["offer_id"] = df["value"].apply(lambda x: x.get("offer_id") or x.get("offer id"))
    df["amount"] = df["value"].apply(lambda x: x.get("amount"))
    df = df.drop(["value"], axis=1)
    
    # merge with portfolio
    return df.sort_index().merge(portfolio_df[["id", "offer_type", "difficulty", "duration"]], left_on="offer_id", right_on="id", how="left")

def process_one_customer_transcript(df):
    """ process transcript table
    """
    # bring duration into this dataframe
    errors = []
    res = []
    for _, row in df.iterrows():
        if row.event == "offer received":
            res.append({"offer_id": row.offer_id, "receive_time": row.time,
                        "amount": 0, "expected_complete_time": row.time + row.duration*24, 
                        "difficulty": row.difficulty, "offer_type": row.offer_type})
        elif row.event == "offer viewed":
            for i in range(len(res)-1, -1, -1):
                if row.offer_id == res[i]["offer_id"] and "view_time" not in res[i] and\
                (("complete_time" not in res[i]) or (row.time <= res[i]["expected_complete_time"])):
                    res[i]["view_time"] = row.time
                    break
            else:
                errors.append(row.offer_id)
        elif row.event == "offer completed":
            for i in range(len(res)-1, -1, -1):
                if row.offer_id == res[i]["offer_id"] and "complete_time" not in res[i]:
                    res[i]["complete_time"] = row.time
                    break
            else:
                errors.append(row.offer_id)
        elif row.event == "transaction":
            for i in range(len(res)):
                if "complete_time" not in res[i] and row.time <= res[i]["expected_complete_time"]:
                    res[i]["amount"] += row.amount
    
    res = pd.DataFrame(res, columns=[
        'person', 'offer_id', 'offer_type', 'difficulty', 'amount', 'receive_time', 'view_time',
        'complete_time', 'expected_complete_time'])
    res["is_in_expected_complete_time"] = res["expected_complete_time"] >= res["complete_time"]
    res["is_enough_amount"] = res["amount"] >= res["difficulty"]
    res["is_view_event"] = res["view_time"].apply(lambda x: not np.isnan(x))
    res["is_complete_event"] = res["complete_time"].apply(lambda x: not np.isnan(x))
    res["is_complete"] = (((res["is_in_expected_complete_time"] | ((res["offer_type"] == "informational") & (res["amount"] > 0))) & res["is_enough_amount"]) | res["is_complete_event"]) & res["is_view_event"]

    return res, errors


def process_transcript(transcript_df):
    res_df = pd.DataFrame(columns=['person', 'offer_id', 'offer_type', 'difficulty', 'amount',
       'receive_time', 'view_time', 'complete_time', 'expected_complete_time',
       'is_in_expected_complete_time', 'is_enough_amount', 'is_view_event',
       'is_complete_event', 'is_complete'])
    
    # record errors
    errors = []
    persons = transcript_df.person.to_list()
    print(f"Total {len(persons)} persons.")
    for i, person in enumerate(persons):
        if i%1000 == 0:
            print(f"Working on {i} person.")

        # process each customer
        df, error = process_one_customer_transcript(transcript_df[transcript_df.person == person])
        # add person info to dataframe
        df.person = person
        # append dataframe
        res_df = pd.concat([res_df, df], axis=0)
        # record error
        errors.append([person, error])

        if len(res_df)>0 and len(res_df)%50000 == 0:
            print(f"Exporting {i}")
            # export users
            res_df.to_csv(f"data/processed_transcript_{i}.csv")
            res_df = pd.DataFrame(columns=['person', 'offer_id', 'offer_type', 'difficulty', 'amount',
            'receive_time', 'view_time', 'complete_time', 'expected_complete_time',
            'is_in_expected_complete_time', 'is_enough_amount', 'is_view_event',
            'is_complete_event', 'is_complete'])

            # export errors
            with open(f"data/errors_{i}.pickle", "wb") as f:
                pickle.dump(errors, f)
            errors = []

    return res_df, errors

if __name__ == "__main__":
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    df = transform_transcript(portfolio)
    processed_df, errors = process_transcript(df)
    processed_df.to_csv("data/processed_transcript_1.csv")

    import pickle
    with open("data/errors_1.pickle", "wb") as f:
        pickle.dump(errors, f)
