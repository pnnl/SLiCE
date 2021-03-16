from src.processing.generic_attributed_graph import GenericGraph

def main():
    dataset = "dblp"
    benchmarks_dir = "/projects/streaming_graph/contextual_embeddings/benchmarks"
    gatne_data_dir = f"{benchmarks_dir}/GATNE/data/"
    compgcn_dir = f"{benchmarks_dir}/CompGCN/data"
    compgcn_data_dir = f"{compgcn_dir}/{dataset}"
    dataset_dir = f"{gatne_data_dir}/{dataset}"
    gtn_data_dir = f"{benchmarks_dir}/Graph_Transformer_Networks/data"
    gtn_dump_dir = f"{gtn_data_dir}/{dataset}"
    false_edge_gen = None
    graph = GenericGraph(dataset_dir, false_edge_gen)
    graph.dump_graph_transformer_networks_dataset(compgcn_data_dir,
                                                  gtn_dump_dir)
    print()


if __name__ == "__main__":
    main()