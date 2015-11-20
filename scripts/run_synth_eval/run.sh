DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

OUTPUT="$DIR/output_log.txt"
ERROR="$DIR/error_log.txt"

rm $OUTPUT
rm $ERROR

(julia -p 31 $DIR/run_synth_eval.jl >> $OUTPUT 2>> $ERROR &)
