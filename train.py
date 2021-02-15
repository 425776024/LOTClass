from config.configs_interface import configs as args
from src.trainer import LOTClassTrainer
from src.logers import LOGS
import os
LOGS.init(os.path.join(args.project.PROJECT_DIR, f'{args.log.log_dir}/{args.log.log_file_name}'))


'''
1.标签名称替换：利用并理解标签名称，通过MLM生成类别词汇；

2.类别预测：通过MLM获取类别指示词汇集合，并构建基于上下文的单词类别预测任务，训练LM模型；

3.自训练：基于上述LM模型，进一步对未标注语料进行自训练后，以更好泛化！
'''


def main():
    trainer = LOTClassTrainer(args)
    # # Construct category vocabulary
    trainer.category_vocabulary(top_pred_num=args.train_args.top_pred_num,
                                category_vocab_size=args.train_args.category_vocab_size)
    # # Training with masked category prediction
    trainer.mcp(top_pred_num=args.train_args.top_pred_num, match_threshold=args.train_args.match_threshold,
                epochs=args.train_args.MCP_EPOCH)
    # # # Self-training
    trainer.self_train(epochs=args.train_args.SELF_TRAIN_EPOCH, loader_name=args.data.final_model)
    # # # Write test set results
    if args.data.TEST_CORPUS is not None:
        trainer.write_results(loader_name=args.data.final_model, out_file=args.data.out_file)


if __name__ == "__main__":
    main()
