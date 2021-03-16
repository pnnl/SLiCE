for dataset in "youtube" "amazon_s" 'freebase' #'twitter'
do
		cd $dataset
		mkdir bfs1 bfs5 bfs10 bfs20 bfs40
		mkdir dfs1 dfs5 dfs10 dfs20 dfs40
		for mydir in dfs*;do mkdir $mydir/$dataset;done
		for mydir in bfs*;do mkdir $mydir/$dataset;done
		for mydir in dfs*;do cd $mydir/$dataset; ln -s ../../*.txt .;cd ../..;done
		for mydir in bfs*;do cd $mydir/$dataset; ln -s ../../*.txt .;cd ../..;done
		cd ../
done
