if [ $# -lt 1 ]; then
    echo "At least one dataset should be specified. Or you can download all datasets by running this script with the argument 'all'."
    exit
fi

declare -A links=( 
    ["mnli"]="https://cloud.tsinghua.edu.cn/f/a7869c56befe4d19b524/?dl=1" 
    ["qqp"]="https://cloud.tsinghua.edu.cn/f/f77e48cdc19e4376a926/?dl=1" 
    ["record"]="https://cloud.tsinghua.edu.cn/f/6994d5960f4846248d1f/?dl=1" 
    ["squad2"]="https://cloud.tsinghua.edu.cn/f/f7bacdc4ec76401a818e/?dl=1"
    ["xsum"]="https://cloud.tsinghua.edu.cn/f/3c824bd90b1a4896a267/?dl=1"
)

download_all_flag=false
for dataset in "$@" 
do
    if [ $dataset = "all" ]; then
        download_all_flag=true
        break
    fi
done

if [ $download_all_flag = true ]; then
    DATASETS=(mnli qqp record squad2 xsum)
else
    DATASETS=($@)
fi

echo "Downloading datasets: ${DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
    echo ${dataset}
    wget --no-check-certificate -O data/${dataset}.tar.gz ${links[$dataset]}
    tar zxvf data/${dataset}.tar.gz -C data
    rm data/${dataset}.tar.gz
done