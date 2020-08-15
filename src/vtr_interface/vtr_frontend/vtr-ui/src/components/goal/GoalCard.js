import clsx from "clsx";
import React from "react";

import Box from "@material-ui/core/Box";
import Button from "@material-ui/core/Button";
import Card from "@material-ui/core/Card";
import ClearIcon from "@material-ui/icons/Clear";
import MoreVertIcon from "@material-ui/icons/MoreVert";
import Typography from "@material-ui/core/Typography";
import { withStyles } from "@material-ui/core/styles";
import { sortableHandle } from "react-sortable-hoc";

const DragHandle = sortableHandle(() => <MoreVertIcon fontSize={"large"} />);

const styles = (theme) => ({});

class GoalCard extends React.Component {
  render() {
    const { active, className, goal, handleClick, removeGoal } = this.props;
    return (
      <Card className={clsx(className)} style={{ width: "100%" }}>
        <Box width={1} display={"flex"} flexDirection={"row"}>
          <Box width={0.1} my={"auto"}>
            <DragHandle></DragHandle>
          </Box>
          <Box width={0.9} display={"flex"} flexDirection={"column"}>
            <Box width={1} display={"flex"} flexDirection={"row"}>
              <Box width={200} mx={1} my={"auto"}>
                <Typography variant="button">{goal.target}</Typography>
              </Box>
              <Box width={100} m={1}>
                <Button
                  color={"primary"}
                  disableElevation={true}
                  fullWidth={true}
                  size="small"
                  startIcon={<ClearIcon />}
                  variant={"contained"}
                  onClick={(e) => removeGoal(goal, e)}
                >
                  Cancel
                </Button>
              </Box>
            </Box>
            <Box display={"flex"} width={1} m={1}>
              <Box display={"flex"} width={0.5} mr={0.5}>
                <Typography variant="button">
                  {"Before:" + goal.pauseBefore.toFixed(1) + "s"}
                </Typography>
              </Box>
              <Box display={"flex"} width={0.5} ml={0.5}>
                <Typography variant="button">
                  {"After:" + goal.pauseAfter.toFixed(1) + "s"}
                </Typography>
              </Box>
            </Box>
            {goal.target === "Repeat" && (
              <Box
                mx={1}
                my={"auto"}
                width={1}
                display={"flex"}
                flexDirection={"row"}
              >
                <Box
                  my={"auto"}
                  width={0.7}
                  mr={0.5}
                  style={{ overflowX: "scroll" }}
                >
                  <Typography variant="button">
                    {"Path:" + goal.path}
                  </Typography>
                </Box>
                <Box width={0.3} ml={0.5} my={1} mr={1}>
                  <Button
                    color={active ? "secondary" : "primary"}
                    disableElevation={true}
                    // variant={"contained"}
                    // // startIcon={<StorageIcon />}
                    size={"small"}
                    onClick={(e) => handleClick()}
                  >
                    {active ? "Clear" : "Show"}
                  </Button>
                </Box>
              </Box>
            )}
          </Box>
        </Box>
      </Card>
    );
  }
}

export default withStyles(styles)(GoalCard);
