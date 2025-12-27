/* ============================================
   RFM 分析與商品推薦 - 精簡版
   ============================================ */

/* 設定 library */
%let path = /home/u64358598/scccupractice;
libname olist "&path";

/* 步驟1: 計算 FM 指標 */
proc sql;
    create table work.fm_data as
    select 
        b.customer_unique_id,
        count(distinct a.order_id) as frequency,
        sum(c.payment_value) as monetary
    from olist.orders as a
    inner join olist.customers as b on a.customer_id = b.customer_id
    inner join olist.order_payments as c on a.order_id = c.order_id
    where a.order_status = 'delivered'
    group by b.customer_unique_id;
quit;

/* 步驟2: 計算中位數並分群 */
proc means data=work.fm_data noprint;
    var frequency monetary;
    output out=work.medians 
        median(frequency)=med_f 
        median(monetary)=med_m;
run;

data _null_;
    set work.medians;
    call symputx('median_f', med_f);
    call symputx('median_m', med_m);
run;

data work.segments;
    set work.fm_data;
    if frequency >= &median_f and monetary >= &median_m then segment = '高價值客戶';
    else if frequency >= &median_f then segment = '高頻客戶';
    else if monetary >= &median_m then segment = '高額客戶';
    else segment = '一般客戶';
run;

/* 步驟3: 選擇目標客戶(高價值) */
data work.target_customers;
    set work.segments;
    where segment = '高價值客戶';
run;

/* 步驟4: 找出目標客戶最愛的產品類別 */
proc sql;
    create table work.popular_categories as
    select 
        e.product_category_name_english as category,
        count(*) as buy_count
    from work.target_customers as a
    inner join olist.customers as b on a.customer_unique_id = b.customer_unique_id
    inner join olist.orders as c on b.customer_id = c.customer_id
    inner join olist.order_items as d on c.order_id = d.order_id
    inner join olist.products as p on d.product_id = p.product_id
    inner join olist.category_translation as e on p.product_category_name = e.product_category_name
    where c.order_status = 'delivered' and e.product_category_name_english ne ''
    group by e.product_category_name_english
    order by buy_count desc;
quit;

/* 步驟5: 取前3名類別 */
data _null_;
    set work.popular_categories(obs=3);
    call symputx(cats('cat',_n_), category);
run;

/* 步驟6: 在熱門類別中找高評分商品 */
proc sql;
    create table work.recommendations as
    select 
        b.product_category_name_english as category,
        a.product_id,
        count(distinct a.order_id) as sales,
        avg(c.review_score) as avg_rating
    from olist.order_items as a
    inner join olist.products as p on a.product_id = p.product_id
    inner join olist.category_translation as b on p.product_category_name = b.product_category_name
    left join olist.order_reviews as c on a.order_id = c.order_id
    where b.product_category_name_english in ("&cat1", "&cat2", "&cat3")
    group by b.product_category_name_english, a.product_id
    having avg(c.review_score) >= 4 and count(distinct a.order_id) >= 5;
quit;

proc sort data=work.recommendations;
    by category descending avg_rating descending sales;
run;

data work.final_recommendations;
    set work.recommendations;
    by category;
    if first.category then rank = 0;
    rank + 1;
    if rank <= 5;
run;

/* 輸出精簡版 Excel */
ods excel file="&path/rfm_result.xlsx";

/* 工作表1: 分群摘要 */
ods excel options(sheet_name='分群摘要');
proc freq data=work.segments;
    tables segment;
    title '客戶分群統計';
run;
title;

/* 工作表2: 目標客戶 TOP 20 */
ods excel options(sheet_name='目標客戶TOP20');
proc print data=work.target_customers(obs=20) noobs;
    var customer_unique_id frequency monetary segment;
    title '高價值客戶 (前20名)';
    format monetary comma12.2;
run;
title;

/* 工作表3: 熱門類別 TOP 10 */
ods excel options(sheet_name='熱門類別TOP10');
proc print data=work.popular_categories(obs=10) noobs;
    title '目標客戶最愛類別 (前10名)';
run;
title;

/* 工作表4: 推薦商品 */
ods excel options(sheet_name='推薦商品');
proc print data=work.final_recommendations noobs;
    var category product_id sales avg_rating rank;
    title '推薦商品清單 (各類TOP 5)';
    format avg_rating 8.2;
run;
title;

ods excel close;

/* 清理暫存檔 */
proc datasets library=work nolist;
    delete fm_data medians segments target_customers 
           popular_categories recommendations;
quit;