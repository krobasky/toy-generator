import argparse


def p(description
      , epochs=3
      , learning_rate=0.0005
      , batch_size=32
      , input_shape=(32,32,3)
      , num_classes=10
      , scale_value=255.0
      , filter=(32,32,64,64)
      , kernel_size=(3,3,3,3)
      , stride=(1,2,1,2)
      , z_dim = 200
      , d_filter=(32,32,64,64)
      , d_kernel_size=(3,3,3,3)
      , d_stride=(1,2,1,2)
      , r_loss_factor=1000
      , print_every_n_batches=100
      , initial_epoch=0
      ):
    
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
    
    _p.add_argument("--filter", '-f',
                    type=int,
                    nargs='+',
                    default=filter,
                    help=f'[{filter}] A list of filters that can be used in the model layers',
                    )

    _p.add_argument("--kernel_size", '-k',
                    type=int,
                    nargs='+',
                    default=kernel_size,
                    help=f'[{kernel_size}] A list of kernels that, with the filters, can be used in the model layers',
                    )

    _p.add_argument("--stride", '-t',
                    type=int,
                    nargs='+',
                    default=stride,
                    help=f'[{stride}] A list of strides that can be used in the model layers',
                    )

    _p.add_argument("--d_filter", '-df',
                    type=int,
                    nargs='+',
                    default=d_filter,
                    help=f'[{d_filter}] A list of filters that can be used in the model layers',
                    )

    _p.add_argument("--d_kernel_size", '-dk',
                    type=int,
                    nargs='+',
                    default=d_kernel_size,
                    help=f'[{d_kernel_size}] A list of kernels that, with the filters, can be used in the model layers',
                    )

    _p.add_argument("--d_stride", '-dt',
                    type=int,
                    nargs='+',
                    default=d_stride,
                    help=f'[{d_stride}] A list of strides that can be used in the model layers',
                    )

    _p.add_argument("--z_dim", "-z`",
                    type=int,
                    default=z_dim,
                    help=f'[{z_dim}] Number of dimensions in the latent space'
                    )
    
    _p.add_argument("--r_loss_factor", '-r',
                    type=int,
                    default=r_loss_factor,
                    help=f'[{r_loss_factor}] factor for weighting loss from lack of normalization vis loss from accuracy.'
                    )

    _p.add_argument("--print_every_n_batches", '-p',
                    type=int,
                    default=print_every_n_batches,
                    help=f'[{print_every_n_batches}] affects verbosity of the training output.'
                    )

    _p.add_argument("--initial_epoch", '-n',
                    type=int,
                    default=initial_epoch,
                    help=f'[{initial_epoch}] which epoch to start on.'
                    )

    
    opts=_p.parse_args()
    return opts
