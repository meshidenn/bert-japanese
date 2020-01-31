import os
import sys
import configparser
import shlex, subprocess
from pathlib import Path


CURDIR = os.path.dirname(os.path.abspath(__file__))


def main(configpath):
    config = configparser.ConfigParser()
    config.read(configpath)
    options = config['BERT-OPTION']
    max_sq_len = options.get('max_sq_length')
    input_dir = Path(options.get('input_file'))
    input_file = ','.join(list(map(str, input_dir.glob('**/all-maxseq{}.tfrecord'.format(max_sq_len)))))
    # input_file = str(input_dir / "AE" / 'all-maxseq{}.tfrecord'.format(max_sq_len))
    output_dir = options.get('output_dir')
    output_dir = os.path.join(output_dir, max_sq_len)
    if not(os.path.exists(output_dir)):
        os.makedirs(output_dir)
    do_train = options.get('do_train')
    do_eval = options.get('do_eval')
    train_batch_size = options.get('train_batch_size')
    max_pre_per_seq = options.get('max_predicitons_per_seq')
    print(max_pre_per_seq)
    # max_pre_per_seq = 20
    num_train_steps = options.get('num_train_steps')
    num_warmup_steps = options.get('num_warmup_steps')
    save_checkpoints_steps = options.get('save_checkpoints_steps')
    learning_rate = options.get('learning_rate')
    albert_config_file = options.get('albert_config_path')
    option = "--input_file={0} --output_dir={1} --do_train={2} --do_eval={3}  --train_batch_size={4} \
  --max_seq_length={5} --max_predictions_per_seq={6} --num_train_steps={7} --num_warmup_steps={8} \
  --save_checkpoints_steps={9} --learning_rate={10} --albert_config_file={11}".format(
      input_file, output_dir, do_train, do_eval, train_batch_size, max_sq_len, max_pre_per_seq, num_train_steps,
      num_warmup_steps, save_checkpoints_steps, learning_rate, albert_config_file)

    if options.get_boolean("train_parallel"):
        raise NotImplementedError("parallel train mode is not implemented on albert")
        
    mycmd = 'python ' + os.path.join(CURDIR, 'ALBERT', 'run_albert_pretraining.py') + " " + option
    p = subprocess.Popen(mycmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.rstrip().decode("utf8"))


if __name__ == '__main__':
    configpath = sys.argv[1]
    main(configpath)
