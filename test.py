class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def rangeQuery(root, l, r, result):
    if root is None:
        return
    
    # 当前节点值在范围内，加入结果集合
    if l <= root.val <= r:
        result.append(root.val)
    
    # 如果当前节点值大于l，递归左子树
    if root.val > l:
        rangeQuery(root.left, l, r, result)
    
    # 如果当前节点值小于r，递归右子树
    if root.val < r:
        rangeQuery(root.right, l, r, result)

def findElementsInRange(root, l, r):
    result = []
    rangeQuery(root, l, r, result)
    return result

# 测试代码
if __name__ == "__main__":
    # 创建示例平衡二叉搜索树
    root = TreeNode(6)
    root.left = TreeNode(2)
    root.right = TreeNode(8)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(4)
    root.left.right.left = TreeNode(3)
    root.left.right.right = TreeNode(5)
    root.right.left = TreeNode(7)
    root.right.right = TreeNode(9)
    
    # 查询范围 [5, 7]
    l = 5
    r = 7
    result = findElementsInRange(root, l, r)
    
    # 输出结果
    print(f"Elements in range [{l}, {r}]: {result}")
