DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(julia $DIR/run_synth_eval.jl >> $DIR/output_log.txt 2>> $DIR/error_log.txt &)