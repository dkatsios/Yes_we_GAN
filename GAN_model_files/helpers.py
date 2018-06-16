from keras.layers import BatchNormalization


def handle_batch_norm(use_batch_norm, is_generator):
    if use_batch_norm is False or (use_batch_norm is True and not is_generator):
        return None

    if isinstance(use_batch_norm, dict):
        use_batch_norm = use_batch_norm['generator'] if is_generator else use_batch_norm['discriminator']

    if isinstance(use_batch_norm, float) or use_batch_norm:
        momentum = use_batch_norm if isinstance(use_batch_norm, float) else 0.8
        return BatchNormalization(momentum=momentum)
