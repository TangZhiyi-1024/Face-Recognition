# 可选：先清空旧聚类模型
Remove-Item ..\data\clustering_gallery.pkl -ErrorAction SilentlyContinue

Set-Location "d:\PycharmProjects\Face-Recognition\src"

# 1) 收集第1个人
python -m cvproj_exc.training --mode cluster --video "..\data\train_data\Alan_Ball\%04d.jpg"

# 2) 收集第2个人
python -m cvproj_exc.training --mode cluster --video "..\data\train_data\Nancy_Sinatra\%04d.jpg"

# 3) 再加第3个人，提高难度
python -m cvproj_exc.training --mode cluster --video "..\data\train_data\Peter_Gilmour\%04d.jpg"


# 4) 测试重识别（看输出 Cluster j 和距离分布）
python -m cvproj_exc.test --mode cluster --video "..\data\test_data\Alan_Ball\%04d.jpg"
python -m cvproj_exc.test --mode cluster --video "..\data\test_data\Nancy_Sinatra\%04d.jpg"
python -m cvproj_exc.test --mode cluster --video "..\data\test_data\Peter_Gilmour\%04d.jpg"
