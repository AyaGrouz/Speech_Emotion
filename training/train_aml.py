from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import os
import argparse
import joblib
import pickle
import json
from train import train_test_split_preprocess, train_model, get_model_metrics


def register_pickled_dataset(
    aml_workspace: Workspace,
    dataset_name: str,
    datastore_name: str,
    file_path: str
) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    files = [(datastore, file_path + "/trainfeatures.pkl"), (datastore, file_path + "/trainlabel.pkl"), (datastore, file_path + "/testfeatures.pkl"), (datastore, file_path + "/testlabel.pkl")]
    dataset = Dataset.File.from_files(files)
    dataset = dataset.register(workspace=aml_workspace,
                               name=dataset_name,
                               create_new_version=True)

    return dataset


def main():
    print("Running train_aml.py")

    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="speech_emotion_model.pkl",
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,a new version of the dataset will be registered"),
        default="speech_dataset",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default="speech_dataset",
    )

    args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    print("Getting training parameters")

    # Load the training parameters from the parameters file
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file")
        train_args = {}

    # Log the training parameters
    print(f"Parameters: {train_args}")
    for (k, v) in train_args.items():
        run.log(k, v)
        #run.parent.log(k, v)
      

 # Get the dataset
    if dataset_name:
      if data_file_path == 'none':
        dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name)  
      else:
        dataset = register_pickled_dataset(run.experiment.workspace,
                                            dataset_name,
                                            "workspaceblobstore",
                                            data_file_path)
    else:
      e = "No dataset provided"
      print(e)
      raise Exception(e) 

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    #run.parent.tag("dataset_id", value=dataset.id)

    # Split the data into test/train
   
    data = train_test_split_preprocess(dataset)

    # Train the model
    model = train_model(data, train_args)

    # Evaluate and log the metrics returned from the train function
    metrics = get_model_metrics(model,data)
    for (k, v) in metrics.items():
        run.log(k, v) 
        #run.parent.log(k, v)

    # Also upload model file to run outputs for history
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    joblib.dump(value=model, filename=output_path)

    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    # upload the model file explicitly into artifacts
    print("Uploading the model into run artifacts...")
    run.upload_file(name="./outputs/models/" + model_name, path_or_stream=output_path)
    print("Uploaded the model {} to experiment {}".format(model_name, run.experiment.name))
    dirpath = os.getcwd()
    print(dirpath)
    print("Following files are uploaded ")
    print(run.get_file_names())

    run.complete()


if __name__ == '__main__':
    main()
