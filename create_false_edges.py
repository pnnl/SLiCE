from src.processing.freebase_attributed_graph import FreebaseGraph

def main():
    #dataset = "FB15k-237"
    dataset = "dblp"
    #dataset = "FB237"
    data_dir = f"/projects/streaming_graph/contextual_embeddings/benchmarks/CompGCN/data/{dataset}"
    #data_dir = f"/projects/streaming_graph/contextual_embeddings/benchmarks/KGEmb/data/{dataset}"
    output_dir = f"/projects/streaming_graph/contextual_embeddings/benchmarks/GATNE/data/{dataset}"
    output_train_path = f"{output_dir}/train.txt"
    output_test_path = f"{output_dir}/test.txt"
    output_valid_path = f"{output_dir}/valid.txt"
    g = FreebaseGraph(data_dir, '')
    # For some reason generate_link_prediction_dataset doesn't actually dump the files
    g.generate_link_prediction_dataset(output_dir, '', '')
    print()
    # Dump generated edges
    g.dump_edges(output_train_path, g.train_edges, False)
    g.dump_edges(output_test_path, g.test_edges, True)
    g.dump_edges(output_valid_path, g.valid_edges, True)
    print()


if __name__ == "__main__":
    main()
