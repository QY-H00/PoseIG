import pickle
import argparse
from data.RHD import RHD_DataReader


def process_evaluation_data(args):
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0
    eval_dataset = RHD_DataReader(path=args.data_root, mode='evaluation', hand_crop=hand_crop,
                                  use_wrist_coord=use_wrist,
                                  sigma=5,
                                  data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
                                  right_hand_flip=hand_flip, crop_size_input=256)
    evaluation_data = list(range(len(eval_dataset)))
    for i in range(len(eval_dataset)):
        print(i, "validation data finished")
        evaluation_data[i] = eval_dataset.__getitem__(i)
    evaluation_data_file = open(f'data_v2.0/processed_data_evaluation.pickle', 'wb')
    pickle.dump(evaluation_data, evaluation_data_file, protocol=pickle.HIGHEST_PROTOCOL)
    evaluation_data_file.close()


# def process_training_data(args):
#     hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0
#     train_dataset = RHD_DataReader(path=args.data_root, mode='training', hand_crop=hand_crop,
#                                    use_wrist_coord=use_wrist,
#                                    sigma=5,
#                                    data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
#                                    right_hand_flip=hand_flip, crop_size_input=256)
#     print("Total train dataset size: {}".format(len(train_dataset)))
#     interval = len(train_dataset) // 20
#     left = len(train_dataset) % 20
#     print("interval", interval, "left", left)
#     training_data = list(range(interval))
#     prev_slot = 0
#     for i in range(len(train_dataset)):
#         slot = i // interval
#         if prev_slot != slot:
#             if slot < 20:
#                 training_data = list(range(interval))
#             else:
#                 training_data = list(range(left))
#         pos = i % interval
#         training_data[pos] = train_dataset.__getitem__(i)
#         print("slot:", slot, "pos:", pos, f"training data {i} finished")
#         prev_slot = slot
#         if pos == interval - 1 or (pos == left - 1 and slot == 20):
#             training_data_file = open(f'data_v2.0/processed_data_training_{slot}.pickle', 'wb')
#             pickle.dump(training_data, training_data_file, protocol=pickle.HIGHEST_PROTOCOL)
#             training_data_file.close()
#             print(f"slot {slot} data finished")

def process_training_data(args):
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uv_sigma = True, True, True, 'small', 12, 180, 0.0
    train_dataset = RHD_DataReader(path=args.data_root, mode='training', hand_crop=hand_crop,
                                   use_wrist_coord=use_wrist,
                                   sigma=5,
                                   data_aug=False, uv_sigma=uv_sigma, rotate=rotate, BL=BL, root_id=root_id,
                                   right_hand_flip=hand_flip, crop_size_input=256)
    print("Total train dataset size: {}".format(len(train_dataset)))
    for i in range(len(train_dataset)):
        training_data_i = train_dataset.__getitem__(i)
        data_i_file = open(f'data_v2.0/processed_training_data/processed_data_training_{i}.pickle', 'wb')
        pickle.dump(training_data_i, data_i_file, protocol=pickle.HIGHEST_PROTOCOL)
        data_i_file.close()
        print(f"training data {i} finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Train Hourglass On 2D Keypoint Detection')
    # Dataset setting
    parser.add_argument(
        '-dr',
        '--data_root',
        type=str,
        default='RHD_published_v2',
        help='dataset root directory'
    )
    # Dataset setting
    parser.add_argument(
        '--process_training_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

    # Dataset setting
    parser.add_argument(
        '--process_evaluation_data',
        default=False,
        action='store_true',
        help='true if the data has been processed'
    )

    args = parser.parse_args()
    if args.process_training_data:
        process_training_data(args)
    if args.process_evaluation_data:
        process_evaluation_data(args)
