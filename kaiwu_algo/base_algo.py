
class BaseAlgo(object):
    def __init__(self, model, config, optimizer=None, device=None, logger=None, monitor=None, **kwargs): 
        """
        算法的初始化方法
        输入参数:
            - model(torch.nn): 神经网络模型
            - optimizer(torch.optimizer): 优化器对象
            - config(Class): 算法配置信息
            - logger(Logger): 日志 
            - monitor(Monitor): 监控
            - kwargs: 所有额外的关键字参数
        返回值:
            无
        """
        self.device = device
        self.model = model
        self.config = config
        self.optimizer = optimizer
        if self.optimizer is not None:
            self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        else:
            self.parameters = None
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        实现算法的核心方法
        输入参数:
            - list_sample_data: list(SampleData) 类型，使用用户自定义的ObsData作为输入进行训练，一般进行批量训练，所以传入SampleData的列表
        返回值:
            无
        """
        
    def calculate_loss(self, list_sample_data, model_output):
        """
        计算 total_loss
        输入参数: 
            - list_sample_data: list(SampleData) 类型，用于计算loss metrics
            list_model_output: list(dict) 类型，模型输出数据，用于计算loss metrics
        返回值:
            - tensor类型，返回loss
        """
     
    def check_sample_data(self, list_sample_data):
        """
        样本数据检查（检查类型是否符合算法要求）
        输入参数:
            - list_sample_data: list(SampleData) 类型
        返回值:
            无
        """
      
    def check_model_output(self, list_model_output):
        """
        模型预测输出结果检查（检查类型是否符合算法要求）。
        输入参数:
            list_model_output: list(dict) 类型
        返回值:
            无
        """

if __name__ == '__main__':
    # Algo = BaseAlgo(1,None)
    pass
