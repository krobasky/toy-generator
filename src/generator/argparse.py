import argparse


def p(description,
      epochs=3,
      learning_rate=0.0005,
      batch_size=32,
      input_shape=(32,32,3),
      num_classes=10,   
      scale_value=255.0):
    
    _p = argparse.ArgumentParser(description=description)
    # xxx add arguments only if they're defaults are passed in
    _p.add_argument("--epochs", "-e",
                    type=int,
                    default=epochs,
                    help=f'[{epochs}] Train for this many epochs'
                    )

    _p.add_argument("--learning_rate", "-l",
                    type=float,
                    default=learning_rate,
                    help=f'[{learning_rate}] Learning rate to use for the model'
                    )

    _p.add_argument("--batch_size", '-b',
                    type=int,
                    default=batch_size,
                    help=f'[{batch_size}] Number of observations used in a batch for each training step'
                    )

    # xxx broken
    _p.add_argument("--quiet", '-q',
                    action='store_true',
                    help=f'Turn off epoch training output'
                    )
    
    _p.add_argument("--input_shape", '-i',
                    type=int,
                    nargs=3,
                    default=input_shape,
                    help=f'[{input_shape}] Over-ride the default 3 target dimension sizes to be used for the input layer',
                    )

    _p.add_argument("--num_classes", "-c",
                    type=int,
                    default=num_classes,
                    help=f'[{num_classes}] Number of classes (labels) in the training dataset'
                    )
    
    _p.add_argument("--scale_value", "-s",
                    type=float,
                    default=scale_value,
                    help=f'[{scale_value}] Divide by this value to normalize all samples'
                    )
    
    opts=_p.parse_args()
    return opts
