### Load required packages
install.packages(c("arules","cluster","e1071","rpart"))
library(arules)
library(cluster)
library(e1071)
library(rpart)

### 1. Customer Segmentation (Mall dataset clustering)
mall <- data.frame(
  CustomerID=1:10,
  Age=c(19,35,26,27,19,40,23,30,35,31),
  Gender=c("Male","Male","Female","Female","Female","Male","Male","Female","Male","Female"),
  AnnualIncome=c(15,26,30,35,40,70,85,60,75,50),
  SpendingScore=c(39,81,6,77,40,76,73,40,60,42)
)
mall_cluster <- kmeans(mall[,c("AnnualIncome","SpendingScore")], centers=5)
plot(mall$AnnualIncome, mall$SpendingScore, col=mall_cluster$cluster,
     main="Mall Customer Segmentation", xlab="Annual Income", ylab="Spending Score")

### 2. Employee dataset clustering
employee <- data.frame(
  EmployeID=c(111,222,333,444,555,666),
  Gender=c("Male","Male","Female","Female","Female","Male"),
  Age=c(28,25,26,25,30,29),
  Salary=c(150000,150000,160000,160000,170000,200000),
  Credit=c(39,27,42,40,64,72)
)
emp_cluster <- kmeans(employee[,c("Salary","Credit")], centers=3)
plot(employee$Salary, employee$Credit, col=emp_cluster$cluster,
     main="Employee Clustering", xlab="Salary", ylab="Credit")

### 3. Naive Bayes vs SVM (diabetes.csv)
diabetes <- read.csv("diabetes.csv")
nb_model <- naiveBayes(Outcome ~ ., data=diabetes)
nb_pred <- predict(nb_model, diabetes)
svm_model <- svm(Outcome ~ ., data=diabetes)
svm_pred <- predict(svm_model, diabetes)
table(nb_pred, diabetes$Outcome)
table(svm_pred, diabetes$Outcome)

### 4. Vegetarian count
veg <- c("yes","yes","yes","no","yes","no","no","yes","yes","yes")
table(veg)

### 5. Scatter plot (mobiles sold vs money)
x <- c(4,1,5,7,10,2,50,25,90,36)
y <- c(12,5,13,19,31,7,153,72,275,110)
plot(x,y, main="Mobiles vs Money", xlab="Mobiles Sold", ylab="Money", col="blue")

### 6. FP-Growth rules (support=50%, confidence=75%)
transactions <- list(
  c("M","O","N","K","E","Y"),
  c("D","O","N","K","E","Y"),
  c("M","A","K","E"),
  c("M","U","C","K","Y"),
  c("C","O","O","K","I","E")
)
trans <- as(transactions, "transactions")
rules_fp <- apriori(trans, parameter=list(supp=0.5, conf=0.75))
inspect(rules_fp)

### 7. Decision Tree vs SVM (diabetes.csv)
dt_model <- rpart(Outcome ~ ., data=diabetes, method="class")
dt_pred <- predict(dt_model, diabetes, type="class")
svm_model2 <- svm(Outcome ~ ., data=diabetes)
svm_pred2 <- predict(svm_model2, diabetes)
table(dt_pred, diabetes$Outcome)
table(svm_pred2, diabetes$Outcome)

### 8. Marks partitioning
marks <- c(55,60,71,63,55,65,50,55,58,59,61,63,65,67,71,72,75)
hist(marks, breaks=3, col="lightblue", main="Equal-Frequency Partitioning")
hist(marks, breaks=seq(min(marks),max(marks),length.out=4), col="orange", main="Equal-Width Partitioning")
marks_cluster <- kmeans(marks, centers=3)
plot(marks, col=marks_cluster$cluster, main="Clustering of Marks")

### 9. Decision tree ARFF example (simplified)
# Example dataset
dt_data <- data.frame(
  Outlook=c("Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"),
  Temp=c("Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"),
  Humidity=c("High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"),
  Wind=c("Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"),
  Play=c("No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No")
)
dt_model2 <- rpart(Play ~ ., data=dt_data, method="class")
plot(dt_model2); text(dt_model2)

### 10. Apriori vs FP-Growth (ARFF dataset)
transactions2 <- list(
  c("SONY","BPL","LG"),
  c("BPL","SAMSUNG"),
  c("BPL","ONIDA"),
  c("SONY","BPL","SAMSUNG"),
  c("SONY","ONIDA"),
  c("BPL","ONIDA"),
  c("SONY","ONIDA"),
  c("SONY","BPL","ONIDA","LG"),
  c("SONY","BPL","ONIDA")
)
trans2 <- as(transactions2, "transactions")
rules_apriori2 <- apriori(trans2, parameter=list(supp=0.2, conf=0.5))
inspect(rules_apriori2)
rules_fp2 <- eclat(trans2, parameter=list(supp=0.2))
inspect(rules_fp2)

### 11. Normalization (strike rates)
sr <- c(100,70,60,90,90)
min_max <- (sr - min(sr)) / (max(sr)-min(sr))
z_score <- (sr - mean(sr)) / sd(sr)
mad_val <- mean(abs(sr - mean(sr)))
z_score_mad <- (sr - mean(sr)) / mad_val
decimal_scaling <- sr / 100
min_max; z_score; z_score_mad; decimal_scaling

### 12. Car AvgSpeed & TotalTime
avg_speed <- c(78,81,82,74,83,82,77,80,70)
total_time <- c(39,37,36,42,35,36,40,38,46)
sd(avg_speed); sd(total_time)
var(avg_speed); var(total_time)

### 13. Apriori & FP-Growth comparison
transactions3 <- list(
  c("M","O","N","K","E","Y"),
  c("D","O","N","K","E","Y"),
  c("M","A","K","E"),
  c("M","U","C","K","Y"),
  c("C","O","O","K","I","E")
)
trans3 <- as(transactions3, "transactions")
rules_apriori3 <- apriori(trans3, parameter=list(supp=0.3, conf=0.7))
inspect(rules_apriori3)
rules_fp3 <- eclat(trans3, parameter=list(supp=0.3))
inspect(rules_fp3)
