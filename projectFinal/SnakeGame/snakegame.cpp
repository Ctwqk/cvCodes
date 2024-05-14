/*
 * Author:  Taiwei Cui
 * Time: 2024-4-18
 * this is the snake game, which can be projected to 3D world.
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <fstream>
#include "CalPos.hpp"

using namespace cv;
using namespace std;

class SnakeGame {
private:
    int width, height;
    deque<pair<int, int>> snake;
    pair<int, int> food;
    int score;
    pair<int, int> dir;
	int step;
	int nodesize;
	int speed;
	int isStarted=0;
	float high=0;
	float initx=0,inity=0;
	deque<vector<Point3f>> snakeBody;
	vector<Point3f> snakeNode;
	vector<Point3f> foodBody;
	vector<Point3f> boundaries; //left,right,upper,lower boundaries

public:
    SnakeGame(int w, int h) : width(w), height(h), score(0), dir({1, 0}) {
        // Initialize snake in the middle
        snake.push_front({w / 2, h / 2});
		//boundaries.push_back(Point3f(initx,inity,high));
		//boundaries.push_back(Point3f(initx,inity+w,high));
		//boundaries.push_back(Point3f(initx+h,inity,high));
		//boundaries.push_back(Point3f(initx+h,inity+w,high));

		boundaries.push_back(Point3f(0,0,0));
		boundaries.push_back(Point3f(0,w,0));
		boundaries.push_back(Point3f(h,0,0));
		boundaries.push_back(Point3f(h,w,0));

        // Place initial food
		step=1;
		nodesize=1;
		speed=10;
		foodBody={
			Point3f(0,1.5,0.5),
			Point3f(2.8532/2,2.9271/2,0.5),
			Point3f(1.7642/2,-2.4271/2,0.5),
			Point3f(-2.2366/2,-2.4271/2,0.5),
			Point3f(-3.1468/2,2.9271/2,0.5),
		};
		snakeNode={Point3f(0.1,0.1,0.1),Point3f(0.9,0.1,0.1),Point3f(0.1,0.9,0.1),Point3f(0.9,0.9,0.1),Point3f(0.1,0.1,0.8),Point3f(0.9,0.1,0.8),Point3f(0.1,0.9,0.8),Point3f(0.9,0.9,0.1)};
		//snakeNode={Point3f(initx+0.1,inity+0.1,high+0.1),Point3f(initx+0.9,inity+0.1,high+0.1),Point3f(initx+0.1,inity+0.9,high+0.1),Point3f(initx+0.9,inity+0.9,high+0.1),Point3f(initx+0.1,inity+0.1,high+0.8),Point3f(initx+0.9,inity+0.1,high+0.8),Point3f(initx+0.1,inity+0.9,high+0.8),Point3f(initx+0.9,inity+0.9,high)};
		snakeBody.push_front(snakeNode);
		for(Point3f &p:snakeBody.front()){
			p.x+=snake.front().first;
			p.y+=snake.front().second;
		}
        placeFood();
    }
	void refresh(Mat &frame,Mat &rvec,Mat &tvec,Mat &camera_matrix,vector<double> &distCoef){
		snakeBody.clear();
		vector<Point3f> tmp;
		vector<Point2f> snake2d;
		for(pair<int,int> p:snake){
			//snakeBody.push_back(Point3f(p.first+initx,p.second+inity,high));
			tmp.clear();
			for(Point3f po:snakeNode){
				tmp.push_back(Point3f(p.first+initx+po.x,p.second+inity+po.y,high+po.z));
			}
			snakeBody.push_back(tmp);
			
		}
		for(Point3f &p:foodBody){
			p.x=food.first+0.5+initx;
			p.y=food.second+0.5+inity;
		}	
		boundaries[0]=Point3f(initx,inity,high);
		boundaries[1]=Point3f(initx,inity+width,high);
		boundaries[2]=Point3f(initx+height,inity,high);
		boundaries[3]=Point3f(initx+height,inity+width,high);
	}

    void placeFood() {
        int fx = rand() % width;
        int fy = rand() % height;
        food = {fx, fy};  // Make sure food is not placed on the snake
		for(Point3f &p:foodBody){
			p.x=fx+0.5+initx;
			p.y=fy+0.5+inity;
		}
    }

    bool updateSnake() {
        pair<int, int> newHead = {snake.front().first + step*dir.first, snake.front().second + step*dir.second};
        // Check boundaries
        if (newHead.first < 0 || newHead.first >= width || newHead.second < 0 || newHead.second >= height) {
            cout << "Boundary! Game Over! Score: " << score << endl;
            return false; // Game over
        }

        // Check self collision
        for (auto segment : snake) {
            if (segment.first == newHead.first && segment.second == newHead.second) {
                cout << "Eat Self! Game Over! Score: " << score << endl;
                return false; // Game over
            }
        }
		snakeBody.push_front(snakeNode);
		for(Point3f &p:snakeBody.front()){
			p.x+=newHead.first+initx;
			p.y+=newHead.second+inity;
			p.z=high;
		}

        snake.push_front(newHead);

        // Eating food
        if (newHead.first-food.first<=nodesize/2&&food.first-newHead.first<=nodesize/2&&newHead.second-food.second<=nodesize/2&&food.second-newHead.second<=nodesize/2) {
            score += 10;
            placeFood();
        } else {
            snake.pop_back(); // Move the snake by removing the tail
			snakeBody.pop_back();
        }

        return true;
    }

    void changeDirection(pair<int, int> newDir) {
        // Prevent the snake from going back on itself
        if ((newDir.first + dir.first != 0) || (newDir.second + dir.second != 0)) {
            dir = newDir;
        }
    }
	void renewImage(Mat &rvec,Mat &tvec,Mat &camera_matrix,vector<double> &distCoef,Mat &frame,int flag=0,int dx=0,int dy=0){
		vector<Point2f> bound2d;
		vector<Point2f> food2d;
		vector<Point2f> snake2d;
		projectPoints(boundaries,rvec,tvec,camera_matrix,distCoef,bound2d);
		for(Point2f p:bound2d){
			circle(frame,p,5,Scalar(255,0,255),-1);
		}
		line(frame,bound2d[0],bound2d[1],Scalar(0,0,255),3,LINE_4);
		line(frame,bound2d[0],bound2d[2],Scalar(0,0,255),3,LINE_4);
		line(frame,bound2d[2],bound2d[3],Scalar(0,0,255),3,LINE_4);
		line(frame,bound2d[1],bound2d[3],Scalar(0,0,255),3,LINE_4);
		projectPoints(foodBody,rvec,tvec,camera_matrix,distCoef,food2d);
		for(Point2f p:food2d){
			circle(frame,p,5,Scalar(0,255,0),-1);
		}
		for(vector<Point3f> v:snakeBody){
			projectPoints(v,rvec,tvec,camera_matrix,distCoef,snake2d);
			for(Point2f p:snake2d){
				circle(frame,p,3,Scalar(255,0,255),-1);
			}
		}
		//cout<<food.first<<" "<<food.second<<endl;
	}

    int runGameLoop(int camera_idx,string cameraParas) {
        Mat gameFrame(height, width, CV_8UC3);
		//cout<<camera_idx-1<<endl;
		ifstream inStream(cameraParas);
		if(!inStream){
			cout<<"failed to read from file"<<endl;
			return -1;
		}
		Mat camera_matrix(3,3,CV_64F);	
		vector<double> distCoef;
		int x,y;
		inStream>>x;
		inStream>>y;
		double *p=camera_matrix.ptr<double>(0);
		for(int i=0;i<9;i++){
			inStream>>p[i];
		}
		double value;
		while(inStream>>value){
			distCoef.push_back(value);
		}
		VideoCapture * vidCap=new VideoCapture(camera_idx);
		if(!vidCap->isOpened()){
			cout<<"failed to open camera"<<endl;
			return -1;
		}
		int framecounter=0;
		Mat rvec,tvec;
		vector<vector<Point3f>>  targets;
		vector<vector<Point2f>>  results;
		vector<Scalar> scalars{Scalar(0,0,255),Scalar(0,255,0),Scalar(255,0,0)};

        for(;;) {
			*vidCap>>gameFrame;
            if (isStarted&&!(framecounter%10)&&!updateSnake()) break; // Update and check for game over
			framecounter++;
            //gameFrame.setTo(Scalar(0, 0, 0)); // Clear screen
			targets=vector<vector<Point3f>>(snakeBody.begin(),snakeBody.end());
			//cout<<targets[0].size()<<endl;
			if(CalPos(gameFrame, tvec,rvec,camera_matrix,distCoef,x,y,targets,results,scalars[0])){
				//cout<<"catch"<<endl;
				putText(gameFrame,"no target found",Point2f(gameFrame.cols/4,gameFrame.rows/2),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"can not deploy play field",Point2f(gameFrame.cols/4,gameFrame.rows/2+40),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"press 'q' to quit",Point2f(gameFrame.cols/4,gameFrame.rows/2+80),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
            	imshow("Snake Game", gameFrame);
				char c=waitKey(5);
				if(c=='q') break;
				continue;
			}
			if(!isStarted){
				putText(gameFrame,"press 'w' 'a' 's' 'd'",Point2f(gameFrame.cols/4,gameFrame.rows/2-100),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"to move play field",Point2f(gameFrame.cols/4,gameFrame.rows/2-60),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"press '<' '>' ",Point2f(gameFrame.cols/4,gameFrame.rows/2-20),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"to up and down",Point2f(gameFrame.cols/4,gameFrame.rows/2+20),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"press 'b' 'c' ",Point2f(gameFrame.cols/4,gameFrame.rows/2+60),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
				putText(gameFrame,"to begin  and cease",Point2f(gameFrame.cols/4,gameFrame.rows/2+100),FONT_HERSHEY_SIMPLEX,1.0,Scalar(255,0,0),2);
					
			}
			renewImage(rvec,tvec,camera_matrix,distCoef,gameFrame);
            imshow("Snake Game", gameFrame);
            char key = waitKey(5); // Delay for 33 ms
			pair<int,int> newDir;
            if (key == 'q'){
				cout<<"User Quit, Score: "<<score<<endl;
				break;
			}	
			else if(key=='b') {
				isStarted=1;
			}
			else if(key=='c'){
				isStarted=0;
			}
			else if (key=='w'){
				if(isStarted)changeDirection({0,1});
				else{
					inity+=0.5;
					refresh(gameFrame,rvec,tvec,camera_matrix,distCoef);
				}
			}
			else if(key=='a'){
				if(isStarted)changeDirection({-1,0});
				else{
					initx-=0.5;
					refresh(gameFrame,rvec,tvec,camera_matrix,distCoef);
				}
			}
			else if(key=='s'){
				if(isStarted)changeDirection({0,-1});
				else{
					inity-=0.5;
					refresh(gameFrame,rvec,tvec,camera_matrix,distCoef);
				}
			}
			else if(key=='d'){
				if(isStarted)changeDirection({1,0});
				else{
					initx+=0.5;
					refresh(gameFrame,rvec,tvec,camera_matrix,distCoef);
				}
			}
			
        }
		return 0;
    }
};
int main(int argc,char** argv) {
    SnakeGame game(20,20);
    game.runGameLoop(atoi(argv[1]),"intcPara.txt");
    return 0;
}
