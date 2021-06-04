# purchase-redeem-forecast
# 数据来源
数据来自[天池大数据](https://tianchi.aliyun.com/competition/entrance/231573/information)

# 数据集构造
1. 不考虑用户特征 <br>
直接将所有用户的数据进行聚合，然后将银行拆解率mfd_day_share_interest和余额宝收益率mdf_bank_shilbor数据时间的匹配，
利用向前填充对缺失数据进行填充。<br>
这里将验证集合目标设置为2014年8月的最后30天（8月2日-8月31日），最终天池比赛的预测目标为2014年9月的最后30天（9月1日-9月30日）