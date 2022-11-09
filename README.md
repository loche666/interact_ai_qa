互动问答智能回复说明
-----------------
本项目分为三个功能模块：数据模块、服务模块和模型模块。

一、数据模块：查询互动问答ES（interact_qa_data）最近1000天的问答数据，调用模型模块获取每个问题的类型和句向量表示，存储到新建的智能回复索引中（intelligent_interact_qa）。
新建索引除问题内容、类型和句向量外，还包含提问公司所属申万和证监会二级行业、板块等信息，用于检索使用。相关文件列表如下：
* 代码入口：construct_es_data.py，若在测试/生产环境执行，需要用DockerfileData打包镜像执行。
>相关命令如下：
>>docker build -f DockerfileData -t harbor.valueonline.cn/ai/construct_data:latest -t harbor.valueonline.cn/ai/construct_data:v`${version}`
>
>>docker run --name constract_data -itd -e profile=dev harbor.valueonline.cn/ai/construct_data:latest

二、服务模块：智能回复web工程，基于用户上传参数和历史问答对检索候选答案，调用模型服务获得句向量表示，基于语义相似度计算获取候选与问题相似性，并进行排序。
* 代码入口：app.py，用于启动服务，若在测试和生产环境运行，需要基于DockerfileApp打包镜像执行。
>相关命令如下：
>>docker build -f DockerfileApp -t harbor.valueonline.cn/ai/interact_qa:latest -t harbor.valueonline.cn/ai/interact_qa:v`${version}`
>
>>docker run --name interact_qa -itd -e profile=dev -p 8601:8501 harbor.valueonline.cn/ai/interact_qa:latest

三、模型模块：目前包含两部分模型：问题类型分类和问题句向量编码。
问题分类模型将问题分为“公司基本状况”、“生产经营情况”等（共有12个类别，目前仅开放“公司基本状况”、“生产经营情况”和“股东及股权变动”三个类别）；
句向量编码模型将问题转换为128维向量（基于sentence-bert对预训练模型进行微调，再基于PCA对bert的768维向量降维）。
* 代码入口：app_model.py，若在测试和生产环境运行，需要基于DockerfileModel打包镜像执行。
>相关命令如下：
>>docker build -f DockerfileModel -t harbor.valueonline.cn/ai/interact_qa_model:latest -t harbor.valueonline.cn/ai/interact_qa_model:v`${version}`
>
>>nvidia-docker run --name interact_qa_model -itd --runtime=nvidia -e profile=dev -p 8600:8500 harbor.valueonline.cn/ai/interact_qa_model:latest

* 训练代码：TBD

* 训练数据：TBD


