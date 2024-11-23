import matplotlib.pyplot as plt

iterations = []
train_loss = []
val_loss = []
with open("logs/log.txt", "r") as f:
  for line in f:
    # print(line.split(" ")[1])
    # Iteration (x-axis)
    line_segments = line.split(" ")
    iterations.append(int (line_segments[4] ))
    train_loss.append(float (line_segments[8] ))

print(len(train_loss))
print(len(iterations))


# Configure the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_loss, label="Train Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

# Save the plot as "loss.png"
# plt.show()
plt.savefig("assets/images/babygpt_training_loss.png")

