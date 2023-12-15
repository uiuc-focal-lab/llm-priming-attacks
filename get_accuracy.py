import csv
import argparse

parser = argparse.ArgumentParser(description="Get accuracy from CSV file")
parser.add_argument("file", help="CSV file to read")

args = parser.parse_args()

with open(args.file, "r") as f:
    reader = csv.reader(f)
    data = list(reader)

    success_count = 0
    total = 0

    for row in data:
        prediction = row[-1]

        if prediction == "yes":
            success_count += 1
        
        total += 1
    
    print("Attack Success Rate: {:.2f}%".format(
        success_count / total * 100))