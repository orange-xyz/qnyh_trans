import sys
from time import time
import tensorflow as tf
import numpy as np

class Transformer_action_model():
    def __init__(self):
        super().__init__()

        self.add_data = True # add_data can't = False
        save_path = 'save/'
        data_path = 'data/task_data/'
        self.test_file = save_path + 'test_file.txt'
        self.grade_file = data_path + 'qnyh_task_data_grades_50k.txt'
        self.class_file = data_path + 'qnyh_task_data_classes_50k.txt'
        self.g_grade_file = data_path + 'g_grades.txt'
        self.g_class_file = data_path + 'g_classes.txt'

        def init_sess():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            return sess
        self.sess = init_sess()

    def train_epoch(self, sess, trainable_model, data_loader, data_loader_grade, data_loader_class):
        supervised_g_losses = []
        data_loader.reset_pointer()
        data_loader_grade.reset_pointer()
        data_loader_class.reset_pointer()

        for it in range(data_loader.num_batch):
            batch = data_loader.next_batch()
            batch_grade = data_loader_grade.next_batch()
            batch_class = data_loader_class.next_batch()
            _, g_loss = trainable_model.pretrain_step(sess, batch, batch_grade, batch_class)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)
    
    def train(self):
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token, add_data = self.add_data)
        self.generator = generator

        self.gen_data_loader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.epoch_num = 30
        self.sess.run(tf.global_variables_initializer())
        self.gen_data_loader.create_batches(self.oracle_file)
        if(self.add_data == True):
            self.data_loader_grade = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
            self.data_loader_class = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
            self.data_loader_grade.create_batches(data_file=self.grade_file)
            self.data_loader_class.create_batches(data_file=self.class_file)

        saver=tf.train.Saver()
        print('start train transformer:')
        for epoch in range(self.epoch_num):
            start = time()
            if(self.add_data == True):
                loss = self.train_epoch(self.sess, self.generator, self.gen_data_loader, self.data_loader_grade, self.data_loader_class)
            else:
                loss = self.train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            print('Now the loss is %.6f'%loss)
            self.add_epoch()
            if epoch % 5 == 0:
                my_generate_samples(self.sess, self.generator, self.batch_size, self.sequence_length, self.generate_num, self.generator_file,
                                    g_grade_file=self.g_grade_file, g_class_file=self.g_class_file)
                #self.evaluate()
                saver.save(self.sess, 'ckpt/lstm.ckpt')

        saver=tf.train.Saver()
        model_file=tf.train.latest_checkpoint('ckpt/')
        saver.restore(self.sess, model_file)
        my_generate_samples(self.sess, self.generator, self.batch_size, self.sequence_length, self.generate_num, self.generator_file,
                            g_grade_file=self.g_grade_file, g_class_file=self.g_class_file)

if __name__ == '__main__':
    TAM = Transformer_action_model()
    TAM.train()