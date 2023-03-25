use pandas_rs::prelude::*;

fn distance(lng1: f64, lat1: f64, lng2: f64, lat2:f64) -> f64 {
    let pi = std::f64::consts::FRAC_PI_4;
    let angle = 0.5 * (lng1 + lng2) * pi / 180.0;
    let mut x_m = 111194.926644 * angle.cos();
    let d_lat = lat1 - lat2;
    x_m = x_m * d_lat.abs();
    let mut y_m = 111194.926644;
    let d_lng = lng1 - lng2;
    y_m = y_m * d_lng.abs();
    let d = x_m*x_m+y_m*y_m;
    return d.sqrt();
}

fn main() {
    let df = Pd::read_csv("D:/OneDrive/桌面/毕设/代码/计及负荷异常增长的空间负荷预测与配电网规划/3.数据集清洗（续）/中压负荷数据表.csv").unwrap(); //read csv file
    let names = Pd::get_column(&df, "年份"); // 负荷名称
    let year = Pd::get_column(&df, "维度Lng"); // 年份
    let lngs = Pd::get_column(&df, "经度Lat"); // 维度Lng
    let lats = Pd::get_column(&df, "区域"); // 经度Lat
    println!("{}", names[0]);
    let radius_m = [[0.0, 100.0], [100.0, 200.0], [200.0, 300.0]];
    let mut r0:Vec<String> = Vec::with_capacity(80000);
    let mut r1:Vec<String> = Vec::with_capacity(80000);
    let mut r2:Vec<String> = Vec::with_capacity(80000);
    for (i, name) in names.iter().enumerate() {
        let year_:i32 =  year[i].parse().unwrap();
        if year_ != 2016 {
            continue;
        }
        r0.push(String::from(""));
        r1.push(String::from(""));
        r2.push(String::from(""));
        if i % 10000 == 0 {
            println!("{}", i);
        }
        // r0.display();
        // r1.display();
        // r2.display();
        for (i_, name_) in names.iter().enumerate() {
            let year_:i32 =  year[i_].parse().unwrap();
            if year_ != 2016 {
                continue;
            }
            let lng1:f64 = lngs[i].parse().unwrap();
            let lat1:f64 = lats[i].parse().unwrap();
            let lng2:f64 = lngs[i_].parse().unwrap();
            let lat2:f64 = lats[i_].parse().unwrap();
            let dis = distance(lng1, lat1, lng2, lat2);
            if dis <= radius_m[0][1] && dis > radius_m[0][0] {
                match r0.get(i) {
                    Some(str) => {
                        if str == "" {
                            r0.push(name_.clone());
                        } else {
                            r0[i].push_str("|");
                            r0[i].push_str(name_);
                        }
                    }
                    None => {r0.push(name_.clone());}
                }
            }
            if dis <= radius_m[1][1] && dis > radius_m[1][0] {
                match r1.get(i) {
                    Some(str) => {
                        if str == "" {
                            r1.push(name_.clone());
                        } else {
                            r1[i].push_str("|");
                            r1[i].push_str(name_);
                        }
                    }
                    None => {r1.push(name_.clone());}
                }
            }
            if dis <= radius_m[2][1] && dis > radius_m[2][0] {
                match r2.get(i) {
                    Some(str) => {
                        if str == "" {
                            r2.push(name_.clone());
                        } else {
                            r2[i].push_str("|");
                            r2[i].push_str(name_);
                        }
                    }
                    None => {r2.push(name_.clone());}
                }
            }

        }
    }

    let mut new_df:Vec<Vec<String>> = Vec::new();
    let mut new_line:Vec<String> = Vec::new();
    new_line.push(String::from("负荷名称"));
    new_line.push(String::from("0至100m以内"));
    new_line.push(String::from("100至200m以内"));
    new_line.push(String::from("200至300m以内"));
    new_df.push(new_line.clone());
    for (i, name) in names.iter().enumerate() {
        let year_:i32 =  year[i].parse().unwrap();
        if year_ != 2016 {
            continue;
        }
        let mut new_line:Vec<String> = Vec::new();
        new_line.push(name.clone());
        new_line.push(r0[i].clone());
        new_line.push(r1[i].clone());
        new_line.push(r2[i].clone());
        new_df.push(new_line.clone());
    }
    Pd::save_csv(new_df.clone(), "./中压负荷相邻数据表.csv");

}
