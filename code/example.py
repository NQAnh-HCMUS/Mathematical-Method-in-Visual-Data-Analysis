import turtle

def draw_fractal_tree(branch_length, t):
    if branch_length > 5:
        t.forward(branch_length)
        t.right(20)
        draw_fractal_tree(branch_length - 15, t)
        t.left(40)
        draw_fractal_tree(branch_length - 15, t)
        t.right(20)
        t.backward(branch_length)

def main():
    screen = turtle.Screen()
    screen.bgcolor("white")
    
    t = turtle.Turtle()
    t.color("green")
    t.speed(0)
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color("brown")
    
    draw_fractal_tree(100, t)
    
    screen.mainloop()

if __name__ == "__main__":
    main()