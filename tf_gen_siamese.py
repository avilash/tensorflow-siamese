import sys, getopt

import tensorflow as tf

usage_str = 'python tf_gen_siamese.py --checkpoint_src=path_to_trained_imagenet_model --checkpoint_target=path_to_target_folder --ws=0/1'


def rename(checkpoint_src, checkpoint_target, ws):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_src):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_src, var_name)

            new_name = var_name

            # Rename the variable
            new_name = "net1/" + var_name
            print('Renaming %s to %s.' % (var_name, new_name))
            var = tf.Variable(var, name=new_name)
            if ws == 0:
                # Create another for different branch
                new_name = "net2/" + var_name
                print('Creating %s to %s.' % (var_name, new_name))
                var2 = tf.Variable(var.initialized_value(), name=new_name)

        # Save the variables
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint_target)


def main(argv):
    checkpoint_src = None
    checkpoint_target = None
    ws = 1

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_src=', 'checkpoint_target=', 'ws='])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_src':
            checkpoint_src = arg
        elif opt == '--checkpoint_target':
            checkpoint_target = arg
        elif opt == '--ws':
            ws = (int)(arg)

    if not checkpoint_src:
        print('Please specify a checkpoint_src. Usage:')
        print(usage_str)
        sys.exit(2)
    if not checkpoint_target:
        print('Please specify a checkpoint_target. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_src, checkpoint_target, ws)


if __name__ == '__main__':
    main(sys.argv[1:])
