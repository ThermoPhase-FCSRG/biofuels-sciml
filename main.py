from src.data.data_processing import DataProcessing as data


def executor():
    data_processing_exec = data()
    data_processing_exec.process_raw_dataset()

    print("Data successfully saved!")


if __name__ == "__main__":
    try:
        executor()

    except Exception as e:
        print(e)
